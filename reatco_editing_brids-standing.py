import torch
from torch import autocast
from diffusers import DDIMScheduler, AutoencoderKL
from tuneavideo.pipelines.pipeline_split_gen import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid, ddim_inversion, read_mask, to_tensors

Model_DIR = 'tune_a_video_model/birds-standing'
model_version = "checkpoints/stable-diffusion-v1-4"
device = "cuda"
unet = UNet3DConditionModel.from_pretrained(Model_DIR, subfolder='unet', torch_dtype=torch.float16).to(device)
scheduler = DDIMScheduler.from_pretrained(model_version, subfolder='scheduler')
pipe = TuneAVideoPipeline.from_pretrained(model_version, unet=unet, scheduler=scheduler, torch_dtype=torch.float16).to(device)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

prompts = [
            "A cat, a rabbit, and a chicken are standing on a wall.",
          ]
maintain_back = True  # if conduct object editing, need to set Ture to maintain background
bbox = [[29, 249, 173, 391], [321, 257, 437, 382], [438, 260, 590, 386]]  # bbox for region of interest
token_indices_list = [
    [2, 5, 9],
]  # align with bbox
use_inv_latent = True
num_samples = 1
guidance_scale = 12.5
num_inference_steps = 50
video_length = 16
window_size = 4  # split short video clip frames
height = 448
width = 768
max_iter_to_alter = 25

video_latents = None
masks = None
if maintain_back:
    # loading video latents from local device
    video_latents = torch.load('data/video_latents_birds-standing.pt').to(torch.float16)
    # generating back mask
    masks = torch.zeros_like(video_latents).to(torch.float16)
    for box in bbox:
        box = [max(round(b / (height / video_latents.shape[3])), 0) for b in box]
        x1, y1, x2, y2 = box
        ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=masks.dtype).to(masks.device)
        masks[:, :, :, y1:y2, x1:x2] = ones_mask

ddim_inv_latent = None
if use_inv_latent:
    print("Obtaining DDIM inv latents")
    with torch.inference_mode():
        ddim_inv_scheduler = DDIMScheduler.from_pretrained(model_version, subfolder='scheduler')
        ddim_inv_scheduler.set_timesteps(num_inference_steps)
        ddim_inv_latents = ddim_inversion(
            pipe, ddim_inv_scheduler, video_latent=torch.load('data/video_latents_birds-standing.pt').to(torch.float16),
            num_inv_steps=num_inference_steps, prompt="")
        ddim_inv_latent = ddim_inv_latents[-1].to(torch.float16)
    torch.cuda.empty_cache()

for prompt, token_indices in zip(prompts, token_indices_list):
    print(prompt, token_indices)
    seeds = [33]
    for seed in seeds:
        print('current seed:', seed)
        g_cuda = torch.Generator(device=device)
        g_cuda.manual_seed(seed)
        if ddim_inv_latent is not None:
            if ddim_inv_latent.size(2) > video_length:  # need split ddim inv to generation
                split_latent, video_list = [], []
                split_video_latent, split_masks = [], []
                for i in range(0, ddim_inv_latent.size(2)):
                    if len(split_latent) < video_length:
                        split_latent.append(ddim_inv_latent[:, :, i, :, :].unsqueeze(2))
                        if maintain_back:
                            split_video_latent.append(video_latents[:, :, i, :, :].unsqueeze(2))
                            split_masks.append(masks[:, :, i, :, :].unsqueeze(2))
                    if len(split_latent) < video_length and i < (ddim_inv_latent.size(2) - 1):
                        continue
                    split_latent = torch.cat(split_latent, dim=2)
                    if maintain_back:
                        split_video_latent = torch.cat(split_video_latent, dim=2)
                        split_masks = torch.cat(split_masks, dim=2)
                    with autocast(device), torch.no_grad():
                        from ptp_utils import AttentionStore, register_attention_control
                        controller = AttentionStore()
                        register_attention_control(pipe, controller)
                        videos = pipe(
                            prompt,
                            latents=split_latent,
                            video_latents=split_video_latent,
                            masks=split_masks,
                            maintain_back=maintain_back,
                            video_length=split_latent.size(2),
                            height=height,
                            width=width,
                            max_iter_to_alter=max_iter_to_alter,
                            num_videos_per_prompt=num_samples,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            bbox=bbox,
                            attention_store=controller,
                            indices_to_alter=token_indices,
                            generator=g_cuda
                        ).videos
                        video_list.append(videos)
                    split_latent = []
                    split_video_latent = []
                    split_masks = []
                videos = torch.cat(video_list, dim=2)
            else:
                with autocast(device), torch.no_grad():
                    from ptp_utils import AttentionStore, register_attention_control
                    controller = AttentionStore()
                    register_attention_control(pipe, controller)
                    videos = pipe(
                        prompt,
                        latents=ddim_inv_latent,
                        video_latents=video_latents,
                        masks=masks,
                        maintain_back=maintain_back,
                        video_length=video_length,
                        window_size=window_size,
                        height=height,
                        width=width,
                        max_iter_to_alter=max_iter_to_alter,
                        num_videos_per_prompt=num_samples,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        bbox=bbox,
                        attention_store=controller,
                        indices_to_alter=token_indices,
                        generator=g_cuda
                    ).videos

        save_dir = "./edited_videos"
        save_path = f"{save_dir}/{prompt}.gif"
        save_videos_grid(videos, save_path, need_mp4=True)

