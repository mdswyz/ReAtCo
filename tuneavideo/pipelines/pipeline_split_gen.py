import numpy as np
import torch
from torch.nn import functional as F
import diffusers
from .pipeline_video import VideoPipeline
from diffusers.utils import deprecate, logging, BaseOutput
from ptp_utils import AttentionStore, aggregate_attention
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from .gaussian_smoothing import GaussianSmoothing
from dataclasses import dataclass, field

logger = logging.get_logger(__name__)


@dataclass
class TuneAVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class TuneAVideoPipeline(VideoPipeline):
    _optional_components = []

    def _compute_max_attention_per_index(self,
                                         attention_maps: torch.Tensor,
                                         indices_to_alter: List[int],
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         normalize_eot: bool = False,
                                         bbox: List[int] = None,
                                         P: float = 0.2,
                                         config=None,
                                         height=None,
                                         width=None,
                                         latents=None,
                                         att_save_path=None,
                                         ) -> List[torch.Tensor]:
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, :, 1:last_idx]  # size is (F,H,W,75)
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values
        max_indices_list_fg = []
        max_indices_list_bg = []

        for i, box in zip(indices_to_alter, bbox):
            video = attention_for_text[:, :, :, i]  # size is (h/32,w/32)
            box = [max(round(b / (height / video.shape[1])), 0) for b in box]
            x1, y1, x2, y2 = box

            # coordinates to masks
            obj_mask = torch.zeros_like(video)
            ones_mask = torch.ones([y2 - y1, x2 - x1], dtype=obj_mask.dtype).to(obj_mask.device)
            obj_mask[:, y1:y2, x1:x2] = ones_mask
            bg_mask = 1 - obj_mask

            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=3).cuda()
                input = F.pad(video.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), mode='reflect')
                video = smoothing(input).squeeze(0).squeeze(0)

            image_fg, image_bg = [], []
            for j, image in enumerate(video):

                # Inner-Region of Object Constraint
                k = (obj_mask[j].sum() * P).long()
                image_fg.append((image * obj_mask[j]).reshape(-1).topk(k)[0].mean().unsqueeze(0))

                # Outer-Region of Object Constraint
                k = (bg_mask[j].sum() * P).long()
                image_bg.append((image * bg_mask[j]).reshape(-1).topk(k)[0].mean().unsqueeze(0))

            max_indices_list_fg.append(torch.cat(image_fg).mean())
            max_indices_list_bg.append(torch.cat(image_bg).mean())

        return max_indices_list_fg, max_indices_list_bg

    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore,
                                                   indices_to_alter: List[int],
                                                   attention_res: list,
                                                   smooth_attentions: bool = False,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   normalize_eot: bool = False,
                                                   bbox: List[int] = None,
                                                   config=None,
                                                   height=None,
                                                   width=None,
                                                   latents=None,
                                                   att_save_path=None,
                                                   ):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)

        max_attention_per_index_fg, max_attention_per_index_bg = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
            bbox=bbox,
            config=config,
            height=height,
            width=width,
            latents=latents,
            att_save_path=att_save_path,
        )
        return max_attention_per_index_fg, max_attention_per_index_bg

    @staticmethod
    def _compute_loss(max_attention_per_index_fg: List[torch.Tensor], max_attention_per_index_bg: List[torch.Tensor],
                      return_losses: bool = False) -> torch.Tensor:
        losses_fg = [max(0, 1. - curr_max) for curr_max in max_attention_per_index_fg]
        losses_bg = [max(0, curr_max) for curr_max in max_attention_per_index_bg]
        loss = sum(losses_fg) + sum(losses_bg)
        if return_losses:
            return max(losses_fg), losses_fg
        else:
            return max(losses_fg), loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        latents = latents - step_size * grad_cond  # using grad to update latent
        return latents

    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           indices_to_alter: List[int],
                                           loss_fg: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           text_input,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           attention_res: list,
                                           smooth_attentions: bool = True,
                                           sigma: float = 0.5,
                                           kernel_size: int = 3,
                                           max_refinement_steps: int = 20,
                                           normalize_eot: bool = False,
                                           bbox: List[int] = None,
                                           config=None,
                                           height=None,
                                           width=None,
                                           ):
        iteration = 0
        target_loss = max(0, 1. - threshold)
        while loss_fg > target_loss:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            max_attention_per_index_fg, max_attention_per_index_bg = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_eot=normalize_eot,
                bbox=bbox,
                config=config,
                height=height,
                width=width,
                )

            loss_fg, losses_fg = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, return_losses=True)

            if loss_fg != 0:
                latents = self._update_latent(latents, loss_fg, step_size)

            with torch.no_grad():
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses_fg])
            except Exception as e:
                print(e)  # catch edge case :)

                low_token = np.argmax(losses_fg)

            if iteration >= max_refinement_steps:
                break

        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        max_attention_per_index_fg, max_attention_per_index_bg = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
            bbox=bbox,
            config=config,
            height=height,
            width=width,
        )
        loss_fg, losses_fg = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg, return_losses=True)
        return loss_fg, latents, max_attention_per_index_fg

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        indices_to_alter: List[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        video_latents: Optional[torch.FloatTensor] = None,
        maintain_back: bool = False,
        maintain_fore: bool = False,
        noise_scheduler: Optional[diffusers.schedulers.scheduling_ddpm.DDPMScheduler] = None,
        flows: Optional[tuple] = None,
        masks: Optional[torch.FloatTensor] = None,
        task_mode: Optional[str] = None,
        bbox: list = None,
        attention_store: AttentionStore = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 3,
        max_iter_to_alter: int = 25,
        thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
        scale_factor: int = 20,
        window_size: int = 4,
        scale_range: Tuple[float, float] = (1., 0.5),
        ddim_inv_latents: Optional[list] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        # Default height and width to unet
        if thresholds is None:
            thresholds = field(default_factory=lambda: {0: 0.05, 10: 0.5, 20: 0.8})
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # define res of attention for control
        attention_res = [height / 32, width / 32]

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode input prompt
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents_dtype = latents.dtype

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        views = get_views(video_length, window_size=window_size, stride=1)
        count = torch.zeros_like(latents)
        value = torch.zeros_like(latents)

        # Denoising loop
        att_save_path = None
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                count.zero_()
                value.zero_()
                for idx, (t_start, t_end) in enumerate(views):
                    latents_view = latents[:, :, t_start:t_end]
                    if maintain_back:
                        video_latents_view = video_latents[:, :, t_start:t_end]
                        masks_view = masks[:, :, t_start:t_end]
                    with torch.enable_grad():
                        latents_view = latents_view.clone().detach().requires_grad_(True)
                        noise_pred_text = self.unet(latents_view, t,
                                                    encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
                        self.unet.zero_grad()
                        # Get max activation value for each subject token
                        max_attention_per_index_fg, max_attention_per_index_bg = self._aggregate_and_get_max_attention_per_token(
                            attention_store=attention_store,
                            indices_to_alter=indices_to_alter,
                            attention_res=attention_res,
                            smooth_attentions=smooth_attentions,
                            sigma=sigma,
                            kernel_size=kernel_size,
                            normalize_eot=False,
                            bbox=bbox,
                            height=height,
                            width=width,
                            latents=latents_view,
                            att_save_path=att_save_path,
                        )

                        loss_fg, loss = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg)
                        # Refinement from attention control (not necessary)
                        if i in thresholds.keys() and loss_fg > 1. - thresholds[i]:
                            del noise_pred_text
                            loss_fg, latents_view, max_attention_per_index_fg = self._perform_iterative_refinement_step(
                                latents=latents_view,
                                indices_to_alter=indices_to_alter,
                                loss_fg=loss_fg,
                                threshold=thresholds[i],
                                text_embeddings=text_embeddings,
                                text_input=None,
                                attention_store=attention_store,
                                step_size=scale_factor * np.sqrt(scale_range[i]),
                                t=t,
                                attention_res=attention_res,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                bbox=bbox,
                                height=height,
                                width=width,
                            )

                        # Perform gradient update
                        if i < max_iter_to_alter:
                            _, loss = self._compute_loss(max_attention_per_index_fg, max_attention_per_index_bg)
                            if loss != 0:
                                latents_view = self._update_latent(latents=latents_view, loss=loss,
                                                              step_size=scale_factor * np.sqrt(scale_range[i]))

                    latent_model_input = torch.cat([latents_view] * 2) if do_classifier_free_guidance else latents_view
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if maintain_back:  # Performer IRJS
                        noise = torch.randn_like(video_latents_view)
                        noisy_video_latents = self.scheduler.add_noise(video_latents_view, noise, t)
                        noisy_video_latents_model_input = torch.cat([noisy_video_latents] * 2) if do_classifier_free_guidance else noisy_video_latents
                        latent_model_input = latent_model_input * masks_view + \
                                             (1 - masks_view) * noisy_video_latents_model_input

                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(dtype=latents_dtype)

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if maintain_back:
                        latents_view = latents_view * masks_view + (
                                      1 - masks_view) * noisy_video_latents
                        latents_view_denoised = self.scheduler.step(noise_pred, t, latents_view, **extra_step_kwargs).prev_sample
                    else:
                        latents_view_denoised = self.scheduler.step(noise_pred, t, latents_view, **extra_step_kwargs).prev_sample
                    value[:, :, t_start:t_end] += latents_view_denoised
                    count[:, :, t_start:t_end] += 1
                latents = torch.where(count > 0, value / count, value)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        video = self.decode_latents(latents)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return TuneAVideoPipelineOutput(videos=video)

def get_views(video_length, window_size=16, stride=4):
    num_blocks_time = (video_length - window_size) // stride + 1
    views = []
    for i in range(num_blocks_time):
        t_start = int(i * stride)
        t_end = t_start + window_size
        views.append((t_start,t_end))
    return views