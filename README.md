# ReAtCo

## Re-Attentional Controllable Video Diffusion Editing, AAAI 2025

[![](https://img.shields.io/badge/ArXiv_Version-2412.11710-red
)](http://arxiv.org/abs/2412.11710)
![](https://img.shields.io/badge/Platform-PyTorch-blue) ![](https://img.shields.io/badge/Language-Python-green)
![](https://img.shields.io/badge/License-Apache_2.0-yellow)

This work proposes a new text-guided video editing framework focusing on controllable video generation and editing, with a particular emphasis on the controllability of the spatial location of multiple foreground objects.

## Video Demos

<table class="center">
  <td><img src="video_demo/dolphins-swimming.gif"></td>
  <td><img src="video_demo/A jellyfish and a goldfish are swimming in the blue ocean..gif"></td>
  <td><img src="video_demo/A turtle and a goldfish are swimming in the blue ocean..gif"></td>
  <td><img src="video_demo/A jellyfish and a octopus are swimming in the blue ocean..gif"></td>
  <tr>
  <td width=25% style="text-align:center;">[Source Video]: Two dolphins are swimming in the blue ocean.</td>
  <td width=25% style="text-align:center;">A jellyfish and a goldfish are swimming in the blue ocean, with the jellyfish is to the left of the goldfish.</td>
  <td width=25% style="text-align:center;">A turtle and a goldfish are swimming in the blue ocean, with the turtle is to the left of the goldfish.</td>
  <td width=25% style="text-align:center;">A jellyfish and a octopus are swimming in the blue ocean, with the jellyfish is to the left of the octopus.</td>
</tr>
</table >

<table class="center">
  <td><img src="video_demo/hares-grazing.gif"></td>
  <td><img src="video_demo/A swan and a hare are grazing in the grass..gif"></td>
  <td><img src="video_demo/A cat and a swan are grazing in the grass..gif"></td>
  <td><img src="video_demo/A cat and a swan are grazing in the yellow meadow..gif"></td>
  <tr>
  <td width=25% style="text-align:center;">[Source Video]: "Two hares are grazing in the grass."</td>
  <td width=25% style="text-align:center;">"A swan and a hare are grazing in the grass, with the swan is to the left of the hare."</td>
  <td width=25% style="text-align:center;">"A cat and a swan are grazing in the grass, with the cat is to the left of the swan."</td>
  <td width=25% style="text-align:center;">"A cat and a swan are grazing in the yellow meadow, with the cat is to the left of the swan."</td>
</tr>
</table >

## Overview Framework of ReAtCo
The main idea of ReAtCo is to refocus the cross-attention activation responses between the edited text prompt and the target video during the denoising stage, resulting in a spatially location-aligned and semantically high-fidelity manipulated video. More details can be found in our [paper](http://arxiv.org/abs/2412.11710).

![](framework.jpg)

## Usage
We now introduce how to run our codes and edit the controllable and desired target videos.

### 1. Requirements

We use the classic Tune-A-Video as the pretrained base video editing model so that the Requirements can follow [Tune-A-Video's publicly available codes](https://github.com/showlab/Tune-A-Video).
**Note:** Due to the latest [xformers](https://github.com/facebookresearch/xformers) requiring PyTorch 2.5.1, we have tested our codes on the latest version with the V100 GPU, and the full environment is reported in **`environment.txt`**

### 2. Pretrained Video Editing Model
Before obtaining the Tune-A-Video editing model, you need to download the pretrained [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) model, which should be placed in the `./checkpoints`.
Then run the following command:
```bash
accelerate launch train_tuneavideo.py --config=configs/dolphins-swimming.yaml
```
And, the pretrained video editing models are saved in `./tune_a_video_model`.

### 3. ReAtCo Video Editing

Generating video latents with the following command:
```bash
python generation_video_latents.py
```
Editing videos with the following command:
```bash
python reatco_editing_dolphins-swimming.py
```
The edited videos are saved in `./edited_videos`.

**Note:** In the script above, the default setting is the **Resource-friendly ReAtCo Paradigm**, which ensures that ReAtCo can edit videos on a consumer-grade GPU (e.g. RTX 4090/3090). More details can be found in the Appendix of our [paper](http://arxiv.org/abs/2412.11710). In particular, we set the `window_size=4` as default, which is compatible with RTX 4090/3090 GPU. If you have sufficient GPU resources and do not want to use the resource-friendly paradigm, please set `window_size=video_length`.

### Citation
If you find the codes helpful in your research or work, please cite the following paper:
```
@article{ReAtCo,
  title={Re-Attentional Controllable Video Diffusion Editing},
  author={Wang, Yuanzhi and Li, Yong and Liu, Mengyi and Zhang, Xiaoya and Liu, Xin and Cui, Zhen and Chan, Antoni B.},
  journal={arXiv preprint arXiv:2412.11710},
  year={2024}
}
```
