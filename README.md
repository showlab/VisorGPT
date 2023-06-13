<div align=center>
<img src="visorgpt_title.png" width="400">
</div>

## Learning Visual Prior via Generative Pre-Training [[Arxiv](http://arxiv.org/abs/2305.13777) [Demo](https://huggingface.co/spaces/szukevin/VISOR-GPT) [Video](https://www.youtube.com/watch?v=8FDoBfxSY8I) [Project](https://sierkinhane.github.io/visor-gpt/)]

<img src="demo.gif" width="1000">

## Updates
- [2023/05/23] Paper is available.
- [2023/05/28] Gradio demo is available.
- [2023/05/30] [Hugging Face demo is available](https://huggingface.co/spaces/szukevin/VISOR-GPT).
- [2023/6/13] Training code and data are available.

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
### Step 2 - Download pre-trained weights
Download [visorgpt](https://drive.google.com/file/d/1Pk4UPNKBMH-0uRLmK5COYTca7FUrN8XY/view?usp=sharing), [controlnet-pose2img](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_openpose.pth), [controlnet-sd](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors), [gligen-bbox2img](https://huggingface.co/gligen/gligen-generation-text-box/blob/main/diffusion_pytorch_model.bin), and put them as follow:
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

### Step 3 - Run demo
```
CUDA_VISIBLE_DEVICES=0 python3 gradio_demo.py
```

## Training
Download the training data from [here](https://drive.google.com/file/d/1VVw7zypNtkiMwJa3exGVZ31XnZCjYU6f/view?usp=sharing) and put it into the directory of `train`. The training requires 8 V100(32GB).
```
# change directory
cd train

# train visorgpt using deepspeed
deepspeed pretrain.py --deepspeed --deepspeed_config models/deepspeed_config.json \
                    --dataset_path visorgpt_dagger_train_seq.pt \
                    --vocab_path models/google_uncased_en_coord_vocab.txt \
                    --config_path models/gpt2/config.json \
                    --output_model_path models/visorgpt_dagger_train_seq.bin \
                    --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
                    --total_steps 200000 --save_checkpoint_steps 10000 --report_steps 100 \
                    --learning_rate 5e-5 --batch_size 16
```

## Inference
```
CUDA_VISIBLE_DEVICES=0 python3 scripts/generate_lm_multiple.py --load_model_path models/models/visorgpt_dagger_train_seq.bin/200000/mp_rank_00_model_states.pt \
                               --vocab_path models/google_uncased_en_coord_vocab.txt \
                               --test_path beginning.txt --prediction_path generated_sentence.txt \
                               --config_path models/gpt2/config.json --seq_length 1024
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