import PIL.Image
import json
import argparse
from share import *
from controlnet import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
from pytorch_lightning import seed_everything
from controlnet.annotator.util import resize_image, HWC3
from controlnet.annotator.openpose import OpenposeDetector, util
from controlnet.cldm.model import create_model, load_state_dict
from controlnet.cldm.ddim_hacked import DDIMSampler
from tqdm import tqdm

# apply_openpose = OpenposeDetector()

def build_control_model(config_path='./ControlNet/controlnet/models/cldm_v15.yaml', 
                        model_path='ckpts/controlnet/control_sd15_openpose.pth'):
    model = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(model_path, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler


def build_controlv11_model(config_path='./ControlNet/controlnet/models/cldm_v15.yaml', 
                           sd_path='ckpts/controlnet/v1-5-pruned-emaonly.safetensors',
                            model_path='ckpts/controlnet/control_sd15_openpose.pth'):
    model = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(sd_path, location='cuda'), strict=False)
    model.load_state_dict(load_state_dict(model_path, location='cuda'), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler


def control_infer(model,
                  ddim_sampler, 
                  keypoint_list, 
                  prompt, 
                  a_prompt='best quality, extremely detailed', 
                  n_prompt='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality', 
                  num_samples=1, 
                  detect_resolution=512, 
                  ddim_steps=20, 
                  guess_mode=False, 
                  strength=1.0, 
                  scale=9.0, 
                  seed=-1, 
                  eta=0.0):

    if not prompt:
        prompt = 'in suit'

    with torch.no_grad():
        # input_image = HWC3(input_image)
        # detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        canvas = np.zeros((detect_resolution, detect_resolution, 3), dtype=np.uint8)# + 50
        detected_map = util.draw_bodypose_gt(canvas, keypoint_list)
        detected_map = HWC3(detected_map)
        H, W, C = detected_map.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8).copy()
        results = [x_samples[i] for i in range(num_samples)]

    return [detected_map] + results


if __name__ == "__main__":

    pass