import torch
from diffusers import DDIMScheduler, AutoencoderKL
import decord
from einops import rearrange

model_version = "checkpoints/stable-diffusion-v1-4"
source_video_path = "data/dolphins-swimming.mp4"
save_path = "data/video_latents_dolphins-swimming.pt"
device = "cuda"
height = 448
width = 768

vr = decord.VideoReader(source_video_path, width=width, height=height)
sample_index = list(range(0, len(vr), 1))[:16]
source_video = vr.get_batch(sample_index).asnumpy()
source_video = torch.tensor(source_video)
source_video = rearrange(source_video, "f h w c -> f c h w")
source_video = (source_video / 127.5 - 1.0)
# Convert videos to latent space
pixel_values = source_video.to(device, dtype=torch.float16).unsqueeze(0)
source_video_length = pixel_values.shape[1]
pixel_values = rearrange(pixel_values,
                         "b f c h w -> (b f) c h w")
vae = AutoencoderKL.from_pretrained(model_version, subfolder="vae")
vae.requires_grad_(False)
vae.to(device, dtype=torch.float16)
video_latents = vae.encode(pixel_values).latent_dist.sample()
video_latents = rearrange(video_latents, "(b f) c h w -> b c f h w",
                    f=source_video_length)
video_latents = video_latents * 0.18215
torch.save(video_latents, save_path)
