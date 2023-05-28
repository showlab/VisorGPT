from share import *
import gradio as gr
import numpy as np
import torch
import re
from PIL import Image
from tqdm import tqdm
from train.scripts.generate_lm_multiple import gen_sequence, build_visorgpt
from utils.seq2coord import gen_cond_mask
from gligen.gligen_inference_box import gligen_infer, build_gligen_model
from controlnet.gradio_pose2image_v2 import control_infer, build_control_model, build_controlv11_model

# init models 
visorgpt_config_path = 'train/models/gpt2/config.json'
visorgpt_model_path = 'demo/ckpts/visorgpt/visorgpt_dagger_ta_tb.pt'
visorgpt_vocab_path = 'train/models/google_uncased_en_coord_vocab.txt'

# control_model_path = 'demo/ckpts/controlnet/control_sd15_openpose.pth'
control_model_path = 'demo/ckpts/controlnet/control_v11p_sd15_openpose.pth' # v1.1
control_sd_path = 'demo/ckpts/controlnet/v1-5-pruned-emaonly.safetensors'
control_model_config = 'demo/ControlNet/controlnet/models/cldm_v15.yaml'

gligen_model_path = 'demo/ckpts/gligen/diffusion_pytorch_model_box.bin'


visorgpt_args, visorgpt_model = build_visorgpt(model_config=visorgpt_config_path,
                                                  model_path=visorgpt_model_path,
                                                  vocab_path=visorgpt_vocab_path)
control_model, ddim_sampler = build_controlv11_model(model_path=control_model_path,
                                                     sd_path=control_sd_path,
                                                  config_path=control_model_config)

# build gligen model
g_model, g_autoencoder, g_text_encoder, g_diffusion, \
    g_config, g_grounding_tokenizer_input = build_gligen_model(ckpt=gligen_model_path)


# maximum number of instances
max_num_keypoint = 16
max_num_bbox = 16
max_num_mask = 8

def generate_sequence(gen_type, 
                        data_type,
                        instance_size,
                        num_instance, 
                        object_name_inbox):

    ctn = True

    if gen_type == 'key point':
        num_keypoint = 18
        if num_instance > max_num_keypoint:
            num_instance = max_num_keypoint

        seq_prompt = '; '.join([gen_type, data_type, instance_size, str(num_instance), str(num_keypoint)]) + ' ; [person'

    elif gen_type == 'box' or gen_type == 'mask':

        if not object_name_inbox.strip():
            if gen_type == 'mask':
                object_name_inbox = "bottle; cup"
            else:
                if data_type == 'object centric':
                    object_name_inbox = "great white shark"
                else:
                    object_name_inbox = "person; frisbee"

        num_keypoint = 0

        if gen_type == 'mask':
            if num_instance > max_num_mask:
                num_instance = max_num_mask
        if gen_type == 'box':
            if num_instance > max_num_bbox:
                num_instance = max_num_bbox

        if data_type == 'object centric':
            num_instance = 1

        objects = ', '.join(object_name_inbox.strip().split(";"))
        seq_prompt = '; '.join([gen_type, data_type, instance_size,
                                str(num_instance), str(num_keypoint)]) + '; ' + objects

        if len(object_name_inbox.split(';')) > num_instance:
            return {
                raw_sequence: gr.update(
                    value="The umber of category names should be less than the number of instances, please try again :)",
                    visible=True)
            }

    print("input prompt: \n", seq_prompt)
    sequence = gen_sequence(visorgpt_args, visorgpt_model, seq_prompt)
    assert isinstance(sequence, list)

    try:
        cond_mask, cond_json = gen_cond_mask(sequence, ctn)
        if gen_type == 'key point':
            ori_sequence = cond_json[2]['sequences'][0][0] + '[SEP]'
        elif gen_type == 'box':
            ori_sequence = cond_json[0]['sequences'][0][0] + '[SEP]'
        elif gen_type == 'mask':
            ori_sequence = cond_json[1]['sequences'][0][0] + '[SEP]'        
    except:
        cond_mask, cond_json = gen_cond_mask(sequence, not ctn)    
        if gen_type == 'key point':
            ori_sequence = cond_json[2]['sequences'][0][0] + '[SEP]'
        elif gen_type == 'box':
            ori_sequence = cond_json[0]['sequences'][0][0] + '[SEP]'
        elif gen_type == 'mask':
            ori_sequence = cond_json[1]['sequences'][0][0] + '[SEP]'        

    ret_img = Image.fromarray(cond_mask)

    if not gen_type == 'mask':
        return {
            result_gallery: [ret_img],
            raw_sequence: gr.update(value=ori_sequence, visible=True),
            images_button: gr.update(visible=True),
            text_container: cond_json,
            sequence_container: ori_sequence
        }
    else:
        return {
            result_gallery: [ret_img],
            raw_sequence: gr.update(value=ori_sequence, visible=True),
            images_button: gr.update(visible=False),
            text_container: cond_json,
            sequence_container: ori_sequence
        }        
    
def add_contents(gen_type, 
                        data_type,
                        instance_size,
                        num_instance, 
                        object_name_inbox,
                        num_continuous_gen,
                        global_seq):

    ctn = True

    if gen_type == 'key point':
        num_keypoint = 18
        seq_prompt = '; '.join([gen_type, data_type, instance_size, str(num_instance), str(num_keypoint)]) + ' ; [person'

        if num_continuous_gen:
            ctn = True
            cur_instance = int(global_seq.split(';')[3].strip())
            new_number = cur_instance + num_continuous_gen
            if new_number > max_num_keypoint:
                new_number = max_num_keypoint

            # prompt type a
            if global_seq.split(';')[5].find('[') == -1:
                global_seq = global_seq.replace('[CLS]', '').replace('[SEP]', '')
                objects = re.findall(re.compile(r'[\[](.*?)[]]', re.S), global_seq)
                objects = ' '.join(['[ person' + x + ']' for x in objects])
                seq_prompt = '; '.join([gen_type, data_type, instance_size, str(new_number), str(num_keypoint), objects])
            # prompt type b
            else:
                global_seq = global_seq.replace('[CLS]', '').replace('[SEP]', '')
                seq_list = global_seq.split(';')
                seq_list[3] = str(new_number)
                seq_prompt = ';'.join(seq_list)

    elif gen_type == 'box' or gen_type == 'mask':
        num_keypoint = 0
        if data_type == 'object centric':
            num_instance = 1
        objects = ', '.join(object_name_inbox.strip().split(";"))
        seq_prompt = '; '.join([gen_type, data_type, instance_size,
                                str(num_instance), str(num_keypoint)]) + '; ' + objects
        if len(object_name_inbox.split(';')) > num_instance:
            return {
                raw_sequence: gr.update(value=f"The umber of category names should be less than the number of instances, please try again :)", visible=True)
            }

        if num_continuous_gen:
            cur_instance = int(global_seq.split(';')[3].strip())
            new_number = cur_instance + num_continuous_gen
            
            if gen_type == 'mask':
                if new_number > max_num_mask:
                    new_number = max_num_mask
            if gen_type == 'box':
                if new_number > max_num_bbox:
                    new_number = max_num_bbox

            # prompt type a
            if global_seq.split(';')[5].find('[') == -1:
                global_seq = global_seq.replace('[CLS]', '').replace('[SEP]', '')
                coords = re.findall(re.compile(r'[\[](.*?)[]]', re.S), global_seq)

                objects = global_seq.split(';')[5].split(',')
                objects = ' '.join(['[ ' + objects[i] + coords[i] + ']' for i in range(len(coords))])

                seq_prompt = '; '.join([gen_type, data_type, instance_size, str(new_number), str(num_keypoint), objects])
            # prompt type b
            else:
                global_seq = global_seq.replace('[CLS]', '').replace('[SEP]', '')
                seq_list = global_seq.split(';')
                seq_list[3] = str(new_number)
                seq_prompt = ';'.join(seq_list)

    # import ipdb;ipdb.set_trace()
    print("input prompt: \n", seq_prompt)
    with torch.no_grad():
        sequence = gen_sequence(visorgpt_args, visorgpt_model, seq_prompt)
        torch.cuda.empty_cache()

    assert isinstance(sequence, list)

    try:
        cond_mask, cond_json = gen_cond_mask(sequence, ctn)
        if gen_type == 'key point':
            ori_sequence = cond_json[2]['sequences'][0][0] + '[SEP]'
        elif gen_type == 'box':
            ori_sequence = cond_json[0]['sequences'][0][0] + '[SEP]'
        elif gen_type == 'mask':
            ori_sequence = cond_json[1]['sequences'][0][0] + '[SEP]'        
    except:
        cond_mask, cond_json = gen_cond_mask(sequence, not ctn)    
        if gen_type == 'key point':
            ori_sequence = cond_json[2]['sequences'][0][0] + '[SEP]'
        elif gen_type == 'box':
            ori_sequence = cond_json[0]['sequences'][0][0] + '[SEP]'
        elif gen_type == 'mask':
            ori_sequence = cond_json[1]['sequences'][0][0] + '[SEP]'        

    ret_img = Image.fromarray(cond_mask)


    if not gen_type == 'mask':
        return {
            result_gallery: [ret_img],
            raw_sequence: gr.update(value=ori_sequence, visible=True),
            images_button: gr.update(visible=True),
            text_container: cond_json,
            sequence_container: ori_sequence
        }
    else:
        return {
            result_gallery: [ret_img],
            raw_sequence: gr.update(value=ori_sequence, visible=True),
            images_button: gr.update(visible=False),
            text_container: cond_json,
            sequence_container: ori_sequence
        }

def generate_images(gen_type, 
                    num_samples, 
                    ddim_steps, 
                    object_prompt,
                    seed,
                    global_text,
                    global_seq):

    if gen_type == 'key point':

        data = global_text[2]['keypoints']
        idx = np.arange(len(data))
        split_idx = list(np.array_split(idx, 1)[0])
        for idx in tqdm(split_idx):
            item = data[idx]
            keypoint_list = []
            for ins in item:
                kv = list(ins.items())[0]
                keypoint = (np.array(kv[1])).tolist()
                keypoint_list.append(keypoint)

        with torch.no_grad():
            ret_img = control_infer(model=control_model,
                                ddim_sampler=ddim_sampler,
                                keypoint_list=keypoint_list,
                                prompt=object_prompt.strip(),
                                num_samples=num_samples,
                                ddim_steps=ddim_steps,
                                seed=seed)
            torch.cuda.empty_cache()
            
    elif gen_type == 'box':

        data = global_text[0]['bboxes']

        with torch.no_grad():                
            ret_img = gligen_infer(model=g_model, 
                               autoencoder=g_autoencoder, 
                               text_encoder=g_text_encoder, 
                               diffusion=g_diffusion, 
                               config=g_config,
                               grounding_tokenizer_input=g_grounding_tokenizer_input,
                               context_prompt=object_prompt.strip(),
                               bbox_lists=data,
                               ddim_steps=ddim_steps,
                               batch_size=num_samples, 
                               seed=seed)
            torch.cuda.empty_cache()

    if not gen_type == 'mask':    
        return {
            result_gallery: ret_img,
            text_container: global_text,
            sequence_container: global_seq
        }
    else:
        return {
            raw_sequence: "sequence to mask is not supported yet :)",
            text_container: global_text,
            sequence_container: global_seq
        }


def object_name_inbox_fn(gen_type):

    if gen_type == 'key point':
        return {
            object_name_inbox: gr.update(visible=False),
            data_type: gr.update(choices=['multiple instances']),
            images_button: gr.update(value='Synthesize images using ControlNet'),
            ddim_steps: gr.update(value=20),
            object_prompt: gr.update(placeholder='in suit'),
            num_instance: gr.update(visible=True, minimum=1, maximum=16, value=2, step=1),
            sequence_container: None
        }
    elif gen_type == 'box':
        return {
            object_name_inbox: gr.update(visible=True, value='person; frisbee'),
            data_type: gr.update(choices=['multiple instances', 'object centric']),
            images_button: gr.update(value='Synthesize images using GLIGEN'),
            ddim_steps: gr.update(value=50),
            object_prompt: gr.update(placeholder='man and frisbee'),
            num_instance: gr.update(visible=True, minimum=1, maximum=16, value=2, step=1),
            sequence_container: None
        }

    elif gen_type == 'mask':
        return {
            object_name_inbox: gr.update(visible=True,
                                         label="MS COCO categories to be generated (separated by semicolon)", value='bottle; cup'),
            data_type: gr.update(choices=['multiple instances']),
            images_button: gr.update(value='Synthesize images using GLIGEN'),
            ddim_steps: gr.update(value=50),
            object_prompt: gr.update(placeholder='bottle and cup'),
            num_instance: gr.update(visible=True, minimum=1, maximum=8, value=2, step=1),
            sequence_container: None
        }


def instance_type_change_fn(data_type):

    if data_type == 'multiple instances':
        return {
            md_title: gr.update(visible=True),
            num_continuous_gen: gr.update(visible=True),
            continuous_btn: gr.update(visible=True),
            object_name_inbox: gr.update(label="MS COCO categories to be generated (separated by semicolon)", value='person; frisbee'),
            object_prompt: gr.update(placeholder='man and frisbee'),
            num_instance: gr.update(visible=True, minimum=1, maximum=16, value=2, step=1),
        }

    elif data_type == 'object centric':
        return {
            md_title: gr.update(visible=False),
            num_continuous_gen: gr.update(visible=False),
            continuous_btn: gr.update(visible=False),
            object_name_inbox: gr.update(label="ImageNet-1K categories to be generated", value='great white shark'),
            object_prompt: gr.update(placeholder='great white shark'),
            num_instance: gr.update(visible=False, value=1),
        }

block = gr.Blocks()
with block:

    text_container = gr.State()
    sequence_container = gr.State()

    gr.Markdown('<div align=center> <img src="file/visorgpt_title_all.jpg" width = "100%" height = "100%" /> </div>')

    with gr.Row():
        with gr.Column():

            gr.Markdown("### Params to generate sequences")
            gen_type = gr.inputs.Dropdown(choices=['key point', 'box', 'mask'], type='value', default='key point', label='Anotation Type')
            data_type = gr.inputs.Dropdown(choices=['multiple instances'], type='value', default='multiple instances', label='Data Type')
            instance_size = gr.inputs.Dropdown(choices=['small', 'medium', 'large'], type='value', default='large', label='Instance Size')
            num_instance = gr.Slider(label="Number of instances per image", minimum=1, maximum=16, value=2, step=1)
            object_name_inbox = gr.Textbox(label="MS COCO categories to be generated (separated by semicolon)", placeholder="person; frisbee", visible=False)
            sequence_button = gr.Button(value="Customize sequential output")


            md_title = gr.Markdown("### Continuous generation (Optional)")
            num_continuous_gen = gr.Slider(label="Add instances to the current scene", minimum=1, maximum=16, value=1, step=1)

            continuous_btn = gr.Button(value="Add instances to the current scene")

            gr.Markdown("### Params to synthesize images")
            object_prompt = gr.Textbox(label="Context Prompt", placeholder="in suit", visible=True)  
            num_samples = gr.Slider(label="Batch Size", minimum=1, maximum=36, value=1, step=1)                
            ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)              
            images_button = gr.Button(value="Synthesize images using ControlNet", visible=False)


        with gr.Column():
            raw_sequence = gr.Textbox(label="Raw Sequence", visible=False)
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto', preview=True)                  
            
    gen_type.change(object_name_inbox_fn, inputs=[gen_type],
                    outputs=[object_name_inbox, data_type, images_button, ddim_steps, object_prompt, num_instance, sequence_container])

    data_type.change(instance_type_change_fn, inputs=[data_type],
                     outputs=[md_title, num_continuous_gen, continuous_btn, object_name_inbox, object_prompt, num_instance])


    ips = [gen_type, data_type, instance_size, num_instance, object_name_inbox]
    sequence_button.click(fn=generate_sequence, inputs=ips, outputs=[result_gallery, raw_sequence, images_button, text_container, sequence_container])

    ips = [gen_type, data_type, instance_size, num_instance, object_name_inbox, num_continuous_gen, sequence_container]
    continuous_btn.click(fn=add_contents, inputs=ips, outputs=[result_gallery, raw_sequence, images_button, text_container, sequence_container])

    ips = [gen_type, num_samples, ddim_steps, object_prompt, seed, text_container, sequence_container]
    images_button.click(fn=generate_images, inputs=ips, outputs=[result_gallery, raw_sequence, text_container, sequence_container])

block.launch(server_name='0.0.0.0', server_port=10086, debug=False, share=False)

