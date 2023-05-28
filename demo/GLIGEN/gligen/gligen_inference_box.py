import argparse
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from gligen.ldm.models.diffusion.ddim import DDIMSampler
from gligen.ldm.models.diffusion.plms import PLMSSampler
import os 
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from gligen.ldm.util import instantiate_from_config
from gligen.trainer import read_official_ckpt, batch_to_device
from gligen.inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip 
from functools import partial
import torchvision.transforms.functional as F
import random
from pytorch_lightning import seed_everything

device = "cuda"
"""
for Object Priors as Sequence Modeling
"""
import cv2



def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1)  # filled
    cv2.putText(img, label, c1, 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    return img

def set_alpha_scale(model, alpha_scale):
    from gligen.ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas

def load_ckpt(ckpt_path):
    
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]
    
    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"]  )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config




def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if input == None:
            return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project( feature, torch.load('projection_matrix').cuda().T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if input == None:
            return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature

def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask

@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=60):
    phrases, images = meta.get("phrases"), meta.get("images")
    images = [None]*len(phrases) if images==None else images 
    phrases = [None]*len(images) if phrases==None else phrases 

    # version = "/apdcephfs/share_1290796/jinhengxie/pretrained/clip-vit-large-patch14_gligen"
    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)
    
    text_features = []
    image_features = []
    for phrase, image in zip(phrases,images):
        text_features.append(  get_clip_feature(model, processor, phrase, is_image=False) )
        image_features.append( get_clip_feature(model, processor, image,  is_image=True) )

    # import ipdb;ipdb.set_trace()
    for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 

    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("text_mask"), max_objs ),
        "image_masks" : image_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("image_mask"), max_objs ),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }

    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_kp(meta, batch=1, max_persons_per_image=8):
    
    points = torch.zeros(max_persons_per_image*17,2)
    idx = 0 
    for this_person_kp in meta["locations"]:
        for kp in this_person_kp:
            points[idx,0] = kp[0]
            points[idx,1] = kp[1]
            idx += 1
    
    # derive masks from points
    masks = (points.mean(dim=1)!=0) * 1 
    masks = masks.float()

    out = {
        "points" : points.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
    }

    return batch_to_device(out, device) 

def build_gligen_model(ckpt="ckpts/gligen/diffusion_pytorch_model_box.bin"):
    # - - - - - prepare models - - - - - #
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(ckpt)

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    return model, autoencoder, text_encoder, diffusion, config, grounding_tokenizer_input

def gligen_infer_all_in_one(model, autoencoder, text_encoder, diffusion, config, grounding_tokenizer_input, 
                            ip_model, ip_autoencoder, ip_text_encoder, ip_diffusion, ip_config, ip_grounding_tokenizer_input, 
                            context_prompt, bbox_lists, batch_size=1, seed=-1):
    
    from copy import deepcopy
    all_boxes = deepcopy(bbox_lists)
    largest_index = 0
    max_area = 0
    for i in range(len(bbox_lists[0])):
        cur_dict = bbox_lists[0][i]
        v = list(cur_dict.values())[0]
        area = (v[2] - v[0]) * (v[3] - v[1])        
        if area > max_area:
            largest_index = i
            max_area = area

    gen_list = [[]]
    gen_list[0].append(bbox_lists[0].pop(largest_index))


    ret_img = gligen_infer_tmp(model, autoencoder, text_encoder, diffusion, config, grounding_tokenizer_input, 
                 context_prompt, all_boxes, gen_list, batch_size, seed)
    # imgs = []
    # imgs.append(ret_img[0])
    for i in range(2, len(ret_img)):
        inpainted_img = gligen_inpaint(ip_model, ip_autoencoder, ip_text_encoder, ip_diffusion, ip_config, 
                                    ip_grounding_tokenizer_input, context_prompt, bbox_lists, ret_img[i], 1, seed)
        ret_img.append(inpainted_img[0])
        ret_img.append(inpainted_img[1])
    return ret_img


def gligen_inpaint(model, autoencoder, text_encoder, diffusion, config, grounding_tokenizer_input, 
                 context_prompt, bbox_lists, input_img, batch_size=1, seed=-1):

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--n_splits", type=int,  default=1, help="")
    parser.add_argument("--which_one", type=int,  default=1, help="")
    args = parser.parse_args()
    args.batch_size = batch_size

    # - - - - - update config from args - - - - - #
    config.update(vars(args))
    config = OmegaConf.create(config)

    # seed
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    # - - - - - sampler - - - - - #
    alpha_generator_func = partial(alpha_generator, type=[0.3, 0.0, 0.7])
    if args.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)
        steps = 250
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)
        steps = 50
    from tqdm import tqdm

    idx = np.arange(len(bbox_lists))
    split_idx = list(np.array_split(idx, 1)[0])

    ret_imgs = []
    with torch.no_grad():
        for idx in tqdm(split_idx):
            item = bbox_lists[idx]
            prompt = ''
            phrases = []
            bbox_list = []

            for ins in item:
                kv = list(ins.items())[0]
                category_name = kv[0]
                bbox = kv[1]
                bbox = [j / 512 for j in bbox]

                prompt += 'a ' + category_name + ', '
                phrases.append('a ' + category_name)
                bbox_list.append(bbox)

            # prompt = prompt[:-2]
            prompt = context_prompt

            meta = dict(
                    input_image = 'placeholder',
                    prompt = prompt,
                    phrases = phrases,
                    locations = bbox_list,
                    alpha_type = [0.3, 0.0, 0.7],
                )

            # - - - - - prepare batch - - - - - #
            batch = prepare_batch(meta, config.batch_size)
            context = text_encoder.encode([meta["prompt"]] * config.batch_size)
            uc = text_encoder.encode(config.batch_size * [""])
            starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)

            # - - - - - inpainting related - - - - - #
            inpainting_mask = z0 = None  # used for replacing known region in diffusion process
            inpainting_extra_input = None  # used as model input
            if "input_image" in meta:
                # inpaint mode
                print('built inpainting model!')
                assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'

                inpainting_mask = draw_masks_from_boxes(batch['boxes'], model.image_size).cuda()

                input_image = F.pil_to_tensor(input_img)
                input_image = (input_image.float().unsqueeze(0).cuda() / 255 - 0.5) / 0.5
                z0 = autoencoder.encode(input_image)

                masked_z = z0 * inpainting_mask
                inpainting_extra_input = torch.cat([masked_z, inpainting_mask], dim=1)

                # - - - - - input for gligen - - - - - #
            grounding_input = grounding_tokenizer_input.prepare(batch)
            input = dict(
                x=starting_noise,
                timesteps=None,
                context=context,
                grounding_input=grounding_input,
                inpainting_extra_input=inpainting_extra_input
            )

            # - - - - - start sampling - - - - - #
            shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)

            samples_fake = sampler.sample(S=steps, shape=shape, input=input, uc=uc, guidance_scale=config.guidance_scale,
                                          mask=inpainting_mask, x0=z0)
            # import ipdb
            # ipdb.set_trace()

            samples_fake = autoencoder.decode(samples_fake)

            category = meta['phrases']
            color = []
            for i in range(len(category)):
                color.append([random.randint(0, 255) for _ in range(3)])

            canvas = np.zeros((512, 512, 3), dtype=np.uint8) + 50
            for i in range(len(category)):
                x1, y1, x2, y2 = bbox_list[i]
                x1, y1, x2, y2 = int(x1*512), int(y1*512), int(x2*512), int(y2*512)
                # sample = plot_one_box([x1, y1, x2, y2], sample, color=color[i], label=' '.join(category[i].split(' ')[1:]))
                canvas = plot_one_box([x1, y1, x2, y2], canvas, color=color[i], label=' '.join(category[i].split(' ')[1:]))
                # cv2.rectangle(sample, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # cv2.putText(sample, f'{category[i]}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            canvas = Image.fromarray(canvas)
            ret_imgs.append(canvas)

            for b in range(samples_fake.shape[0]):
                sample = samples_fake[b]
                sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
                sample = (sample.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8).copy()

                sample = Image.fromarray(sample)
                ret_imgs.append(sample)
    return ret_imgs

def gligen_infer_tmp(model, autoencoder, text_encoder, diffusion, config, grounding_tokenizer_input, 
                 context_prompt, all_bboxes, bbox_lists, batch_size=1, seed=-1):

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--n_splits", type=int,  default=1, help="")
    parser.add_argument("--which_one", type=int,  default=1, help="")
    args = parser.parse_args()
    args.batch_size = batch_size

    # - - - - - update config from args - - - - - #
    config.update(vars(args))
    config = OmegaConf.create(config)

    # seed
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    # - - - - - sampler - - - - - #
    alpha_generator_func = partial(alpha_generator, type=[0.3, 0.0, 0.7])
    if args.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)
        steps = 250
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)
        steps = 50
    from tqdm import tqdm

    idx = np.arange(len(bbox_lists))
    split_idx = list(np.array_split(idx, 1)[0])

    ret_imgs = []
    with torch.no_grad():
        for idx in range(len(all_bboxes)):
            item = all_bboxes[idx]
            prompt = ''
            phrases = []
            bbox_list = []

            for ins in item:
                kv = list(ins.items())[0]
                category_name = kv[0]
                bbox = kv[1]
                bbox = [j / 512 for j in bbox]

                prompt += 'a ' + category_name + ', '
                phrases.append('a ' + category_name)
                bbox_list.append(bbox)

            # prompt = prompt[:-2]
            prompt = context_prompt

            meta = dict(
                    prompt = prompt,
                    phrases = phrases,
                    locations = bbox_list,
                    alpha_type = [0.3, 0.0, 0.7],
                )

            category = meta['phrases']
            color = []
            for i in range(len(category)):
                color.append([random.randint(0, 255) for _ in range(3)])

            canvas = np.zeros((512, 512, 3), dtype=np.uint8) + 50
            for i in range(len(category)):
                x1, y1, x2, y2 = bbox_list[i]
                x1, y1, x2, y2 = int(x1*512), int(y1*512), int(x2*512), int(y2*512)
                # sample = plot_one_box([x1, y1, x2, y2], sample, color=color[i], label=' '.join(category[i].split(' ')[1:]))
                canvas = plot_one_box([x1, y1, x2, y2], canvas, color=color[i], label=' '.join(category[i].split(' ')[1:]))
                # cv2.rectangle(sample, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # cv2.putText(sample, f'{category[i]}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            canvas = Image.fromarray(canvas)
            ret_imgs.append(canvas)

        for idx in tqdm(split_idx):
            item = bbox_lists[idx]
            prompt = ''
            phrases = []
            bbox_list = []

            for ins in item:
                kv = list(ins.items())[0]
                category_name = kv[0]
                bbox = kv[1]
                bbox = [j / 512 for j in bbox]

                prompt += 'a ' + category_name + ', '
                phrases.append('a ' + category_name)
                bbox_list.append(bbox)

            # prompt = prompt[:-2]
            prompt = context_prompt

            meta = dict(
                    prompt = prompt,
                    phrases = phrases,
                    locations = bbox_list,
                    alpha_type = [0.3, 0.0, 0.7],
                )

            # - - - - - prepare batch - - - - - #
            batch = prepare_batch(meta, config.batch_size)
            context = text_encoder.encode([meta["prompt"]] * config.batch_size)
            uc = text_encoder.encode(config.batch_size * [""])
            starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)

            # - - - - - inpainting related - - - - - #
            inpainting_mask = z0 = None  # used for replacing known region in diffusion process
            inpainting_extra_input = None  # used as model input
            if "input_image" in meta:
                # inpaint mode
                assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'

                inpainting_mask = draw_masks_from_boxes(batch['boxes'], model.image_size).cuda()

                input_image = F.pil_to_tensor(Image.open(meta["input_image"]).convert("RGB").resize((512, 512)))
                input_image = (input_image.float().unsqueeze(0).cuda() / 255 - 0.5) / 0.5
                z0 = autoencoder.encode(input_image)

                masked_z = z0 * inpainting_mask
                inpainting_extra_input = torch.cat([masked_z, inpainting_mask], dim=1)

                # - - - - - input for gligen - - - - - #
            grounding_input = grounding_tokenizer_input.prepare(batch)
            input = dict(
                x=starting_noise,
                timesteps=None,
                context=context,
                grounding_input=grounding_input,
                inpainting_extra_input=inpainting_extra_input
            )

            # - - - - - start sampling - - - - - #
            shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)

            samples_fake = sampler.sample(S=steps, shape=shape, input=input, uc=uc, guidance_scale=config.guidance_scale,
                                          mask=inpainting_mask, x0=z0)
            # import ipdb
            # ipdb.set_trace()

            samples_fake = autoencoder.decode(samples_fake)

            # os.makedirs(output_folder + '/' + meta['save_folder_name'], exist_ok=True)
            category = meta['phrases']
            color = []
            for i in range(len(category)):
                color.append([random.randint(0, 255) for _ in range(3)])

            canvas = np.zeros((512, 512, 3), dtype=np.uint8) + 50
            for i in range(len(category)):
                x1, y1, x2, y2 = bbox_list[i]
                x1, y1, x2, y2 = int(x1*512), int(y1*512), int(x2*512), int(y2*512)
                # sample = plot_one_box([x1, y1, x2, y2], sample, color=color[i], label=' '.join(category[i].split(' ')[1:]))
                canvas = plot_one_box([x1, y1, x2, y2], canvas, color=color[i], label=' '.join(category[i].split(' ')[1:]))
                # cv2.rectangle(sample, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # cv2.putText(sample, f'{category[i]}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            canvas = Image.fromarray(canvas)
            ret_imgs.append(canvas)

            for b in range(samples_fake.shape[0]):
                sample = samples_fake[b]
                sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
                sample = (sample.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8).copy()

                sample = Image.fromarray(sample)
                ret_imgs.append(sample)
    return ret_imgs


def gligen_infer(model, autoencoder, text_encoder, diffusion, config, grounding_tokenizer_input, 
                 context_prompt, bbox_lists, ddim_steps, batch_size=4, seed=-1):

    parser = argparse.ArgumentParser()
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--n_splits", type=int,  default=1, help="")
    parser.add_argument("--which_one", type=int,  default=1, help="")
    args = parser.parse_args()
    args.batch_size = batch_size

    # - - - - - update config from args - - - - - #
    config.update(vars(args))
    config = OmegaConf.create(config)

    # seed
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    # - - - - - sampler - - - - - #
    alpha_generator_func = partial(alpha_generator, type=[0.3, 0.0, 0.7])
    # if args.no_plms:
    #     sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
    #                           set_alpha_scale=set_alpha_scale)
    #     steps = 250
    # else:
    sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                            set_alpha_scale=set_alpha_scale)
    # steps = 50
    from tqdm import tqdm

    idx = np.arange(len(bbox_lists))
    split_idx = list(np.array_split(idx, 1)[0])

    ret_imgs = []
    with torch.no_grad():
        for idx in tqdm(split_idx):
            item = bbox_lists[idx]
            prompt = ''
            phrases = []
            bbox_list = []

            for ins in item:
                kv = list(ins.items())[0]
                category_name = kv[0]
                bbox = kv[1]
                bbox = [j / 512 for j in bbox]

                prompt += 'a ' + category_name + ', '
                phrases.append('a ' + category_name)
                bbox_list.append(bbox)

            # prompt = prompt[:-2]
            prompt = context_prompt

            meta = dict(
                    prompt = prompt,
                    phrases = phrases,
                    locations = bbox_list,
                    alpha_type = [0.3, 0.0, 0.7],
                )

            # - - - - - prepare batch - - - - - #
            batch = prepare_batch(meta, config.batch_size)
            context = text_encoder.encode([meta["prompt"]] * config.batch_size)
            uc = text_encoder.encode(config.batch_size * [""])
            starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)

            # - - - - - inpainting related - - - - - #
            inpainting_mask = z0 = None  # used for replacing known region in diffusion process
            inpainting_extra_input = None  # used as model input
            if "input_image" in meta:
                # inpaint mode
                assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'

                inpainting_mask = draw_masks_from_boxes(batch['boxes'], model.image_size).cuda()

                input_image = F.pil_to_tensor(Image.open(meta["input_image"]).convert("RGB").resize((512, 512)))
                input_image = (input_image.float().unsqueeze(0).cuda() / 255 - 0.5) / 0.5
                z0 = autoencoder.encode(input_image)

                masked_z = z0 * inpainting_mask
                inpainting_extra_input = torch.cat([masked_z, inpainting_mask], dim=1)

                # - - - - - input for gligen - - - - - #
            grounding_input = grounding_tokenizer_input.prepare(batch)
            input = dict(
                x=starting_noise,
                timesteps=None,
                context=context,
                grounding_input=grounding_input,
                inpainting_extra_input=inpainting_extra_input
            )

            # - - - - - start sampling - - - - - #
            shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)

            samples_fake = sampler.sample(S=ddim_steps, shape=shape, input=input, uc=uc, guidance_scale=config.guidance_scale,
                                          mask=inpainting_mask, x0=z0)
            # import ipdb
            # ipdb.set_trace()

            samples_fake = autoencoder.decode(samples_fake)

            # os.makedirs(output_folder + '/' + meta['save_folder_name'], exist_ok=True)
            category = meta['phrases']
            color = []
            for i in range(len(category)):
                color.append([random.randint(0, 255) for _ in range(3)])

            canvas = np.zeros((512, 512, 3), dtype=np.uint8) + 50
            for i in range(len(category)):
                x1, y1, x2, y2 = bbox_list[i]
                x1, y1, x2, y2 = int(x1*512), int(y1*512), int(x2*512), int(y2*512)
                # sample = plot_one_box([x1, y1, x2, y2], sample, color=color[i], label=' '.join(category[i].split(' ')[1:]))
                canvas = plot_one_box([x1, y1, x2, y2], canvas, color=color[i], label=' '.join(category[i].split(' ')[1:]))
                # cv2.rectangle(sample, (x1, y1), (x2, y2), (0, 255, 0), 3)
                # cv2.putText(sample, f'{category[i]}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            canvas = Image.fromarray(canvas)
            ret_imgs.append(canvas)

            for b in range(samples_fake.shape[0]):
                sample = samples_fake[b]
                sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
                sample = (sample.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8).copy()

                sample = Image.fromarray(sample)
                ret_imgs.append(sample)
    return ret_imgs





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,  default="generation_samples", help="root folder for output")
    parser.add_argument("--sub_folder", type=str,  default="PPSM", help="root folder for output")
    parser.add_argument("--batch_size", type=int, default=1, help="This will overwrite the one in yaml.")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("--n_splits", type=int,  default=1, help="")
    parser.add_argument("--which_one", type=int,  default=1, help="")
    args = parser.parse_args()

    import json
    # read bbox from the pre-prepared .json file
    # with open('/apdcephfs/share_1290796/jinhengxie/code/llm_anno/masterpiece-sier/data/xjh_coco_train.json', 'r', encoding='utf8') as fp:
    # with open('../llm_data_generation/generated_sentence_0.json', 'r', encoding='utf8') as fp:
    # with open('../llm_data_generation/generated_sentence_prompts_for_long_tail_0.json', 'r', encoding='utf8') as fp:
    # with open('../llm_data_generation/generated_sentence_prompts_for_long_tail_0.json', 'r', encoding='utf8') as fp:
    # with open('../llm_data_generation/eval_sentence_0.json', 'r', encoding='utf8') as fp:
    # with open('../llm_data_generation/generated_sentence_demo_0.json', 'r', encoding='utf8') as fp:
    with open('./llm_data_generation/generated_sentence_demo_box.json', 'r', encoding='utf8') as fp:
    # with open('../llm_data_generation/generated_sentence_coco_2_0.json', 'r', encoding='utf8') as fp:
        data = json.load(fp)['bboxes']

    idx = np.arange(len(data))
    split_idx = list(np.array_split(idx, args.n_splits)[args.which_one - 1])

    # - - - - - prepare models - - - - - #
    ckpt = "GLIGEN/ckpts/diffusion_pytorch_model_box.bin"
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(ckpt)

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input
    # - - - - - update config from args - - - - - #
    config.update(vars(args))
    config = OmegaConf.create(config)

    # - - - - - sampler - - - - - #
    alpha_generator_func = partial(alpha_generator, type=[0.3, 0.0, 0.7])
    # import ipdb
    # ipdb.set_trace()
    if args.no_plms:
        sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)
        steps = 250
    else:
        sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func,
                              set_alpha_scale=set_alpha_scale)
        steps = 50
        # - - - - - save - - - - - #
    output_folder = os.path.join(args.folder, args.sub_folder)
    os.makedirs(output_folder, exist_ok=True)
    from tqdm import tqdm

    # import ipdb
    # ipdb.set_trace()
    with torch.no_grad():
        for idx in tqdm(split_idx):
            item = data[idx]
            prompt = ''
            phrases = []
            bbox_list = []

            for ins in item:
                kv = list(ins.items())[0]
                category_name = kv[0]
                bbox = kv[1]
                bbox = [j / 512 for j in bbox]

                prompt += 'a ' + category_name + ', '
                phrases.append('a ' + category_name)
                bbox_list.append(bbox)

            prompt = prompt[:-2]
            # import ipdb
            # ipdb.set_trace()

            # import ipdb
            # ipdb.set_trace()
            # continue
            meta = dict(
                    prompt = prompt,
                    phrases = phrases,
                    locations = bbox_list,
                    alpha_type = [0.3, 0.0, 0.7],
                    save_folder_name=args.sub_folder
                )

            # - - - - - prepare batch - - - - - #
            batch = prepare_batch(meta, config.batch_size)
            context = text_encoder.encode([meta["prompt"]] * config.batch_size)
            uc = text_encoder.encode(config.batch_size * [""])
            starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)

            # - - - - - inpainting related - - - - - #
            inpainting_mask = z0 = None  # used for replacing known region in diffusion process
            inpainting_extra_input = None  # used as model input
            if "input_image" in meta:
                # inpaint mode
                assert config.inpaint_mode, 'input_image is given, the ckpt must be the inpaint model, are you using the correct ckpt?'

                inpainting_mask = draw_masks_from_boxes(batch['boxes'], model.image_size).cuda()

                input_image = F.pil_to_tensor(Image.open(meta["input_image"]).convert("RGB").resize((512, 512)))
                input_image = (input_image.float().unsqueeze(0).cuda() / 255 - 0.5) / 0.5
                z0 = autoencoder.encode(input_image)

                masked_z = z0 * inpainting_mask
                inpainting_extra_input = torch.cat([masked_z, inpainting_mask], dim=1)

                # - - - - - input for gligen - - - - - #
            grounding_input = grounding_tokenizer_input.prepare(batch)
            input = dict(
                x=starting_noise,
                timesteps=None,
                context=context,
                grounding_input=grounding_input,
                inpainting_extra_input=inpainting_extra_input
            )

            # - - - - - start sampling - - - - - #
            shape = (config.batch_size, model.in_channels, model.image_size, model.image_size)

            samples_fake = sampler.sample(S=steps, shape=shape, input=input, uc=uc, guidance_scale=config.guidance_scale,
                                          mask=inpainting_mask, x0=z0)
            # import ipdb
            # ipdb.set_trace()

            samples_fake = autoencoder.decode(samples_fake)

            # os.makedirs(output_folder + '/' + meta['save_folder_name'], exist_ok=True)
            category = meta['phrases']
            color = []
            for i in range(len(category)):
                color.append([random.randint(0, 255) for _ in range(3)])

            for b in range(samples_fake.shape[0]):
                sample = samples_fake[b]
                sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
                sample = (sample.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8).copy()
                canvas = np.zeros((512, 512, 3), dtype=np.uint8) + 50
                for i in range(len(category)):
                    x1, y1, x2, y2 = bbox_list[i]
                    x1, y1, x2, y2 = int(x1*512), int(y1*512), int(x2*512), int(y2*512)
                    # sample = plot_one_box([x1, y1, x2, y2], sample, color=color[i], label=' '.join(category[i].split(' ')[1:]))
                    canvas = plot_one_box([x1, y1, x2, y2], canvas, color=color[i], label=' '.join(category[i].split(' ')[1:]))
                    # cv2.rectangle(sample, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # cv2.putText(sample, f'{category[i]}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                sample = Image.fromarray(sample)
                sample.save(os.path.join(output_folder,  f'{idx}_{b}.png'))
                canvas = Image.fromarray(canvas)
                canvas.save(os.path.join(output_folder, f'{idx}_canvas.png'))



