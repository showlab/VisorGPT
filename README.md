<div align=center>
<img src="visorgpt_title.png" width="400">
</div>

## Learning Visual Prior via Generative Pre-Training [[Arxiv](http://arxiv.org/abs/2305.13777)] [[Demo]()] [[Video](https://www.youtube.com/watch?v=8FDoBfxSY8I)]

## Updates
- Gradio demo is available.
- [Hugging Face demo will be available]().

## Quick Start
### Step 1
```
# clone the repo
git clone https://github.com/Sierkinhane/VisorGPT.git

# go to directory
cd VisorGPT

# create a new environment
conda create -n visorgpt python=3.8

# activate the new environment
conda activate visorgpt

# prepare the basic environments
pip3 install -r requirements.txt

# install controlnet and gligen
cd demo/ControlNet
pip3 install -v -e .
cd ../demo/GLIGEN
pip3 install -v -e .
```
### Step 2 - Download pre-trained weights.
Download [visorgpt](https://drive.google.com/file/d/1Pk4UPNKBMH-0uRLmK5COYTca7FUrN8XY/view?usp=share_link), [controlnet-pose2img](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_openpose.pth), [controlnet-sd](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors), [gligen-bbox2img](https://huggingface.co/gligen/gligen-generation-text-box/blob/main/diffusion_pytorch_model.bin), and put them as follow:
```
├── demo/
|   ├── ckpts
|   |   ├── controlnet
|   |   |   ├── control_v11p_sd15_openpose.pth
|   |   |   ├── v1-5-pruned-emaonly.safetensors
|   |   ├── gligen
|   |   |   ├── diffusion_pytorch_model_box.bin
|   |   ├── visorgpt
|   |   |   ├── visorgpt_dagger_ta_tb.pt
```

### Step 3 - Run demo.
```
CUDA_VISIBLE_DEVICES=0 python3 gradio_demo.py
```

If you are using our code, please consider citing our paper.

```
@article{xie2023visorgpt,
  title={VisorGPT: Learning Visual Prior via Generative Pre-Training},
  author={Xie, Jinheng and Ye, Kai and Li, Yudong and Li, Yuexiang and Lin, Kevin Qinghong and Zheng, Yefeng and Shen, Linlin and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2305.13777},
  year={2023}
}
```