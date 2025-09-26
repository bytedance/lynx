# Copyright (c) 2025 The Wan Team and The HuggingFace Team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by Bytedance Ltd. and/or its affiliates on September 15, 2025.
#
# Original file was released under Apache License 2.0, with the full license text
# available at https://github.com/huggingface/diffusers/blob/v0.30.3/LICENSE and https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

from typing import Any, Callable, Dict, List, Optional, Union

import os
import numpy as np
import torch

from safetensors.torch import load_file
from diffusers.callbacks import MultiPipelineCallbacks
from diffusers.utils import logging, replace_example_docstring
from diffusers import WanPipeline
from modules.full.attention_processor import register_ip_adapter, register_ref_adapter
from modules.common import model_utils_wan as model_utils

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.pipelines.wan.pipeline_wan import *
from modules.full.transformer_wan import WanTransformer3DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class LynxWanPipeline(WanPipeline):
    r""" Pipeline for WanPipeline """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )
        vae = AutoencoderKLWan.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )
        transformer = WanTransformer3DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="transformer"
        )
        scheduler_uni = UniPCMultistepScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )

        text_encoder.requires_grad_(False)
        transformer.requires_grad_(False)
        vae.requires_grad_(False)

        text_encoder.eval()
        transformer.eval()
        vae.eval()

        weight_dtype = kwargs['torch_dtype']
        text_encoder.to(dtype=weight_dtype)
        transformer.to(dtype=weight_dtype)
        vae.to(dtype=weight_dtype)

        pipe = cls(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler_uni,
        )
        return pipe

    def init_image_proj_modules(self, model_dir, device='cuda', dtype=torch.bfloat16):
        from ..common.resampler import Resampler
        self.resampler = Resampler(
            depth=4,
            dim=1280,
            dim_head=64,
            embedding_dim=512,
            ff_mult=4,
            heads=20,
            num_queries=16,
            output_dim=5120,
        )
        self.resampler.to(device=device, dtype=dtype)
        self.resampler.eval()

        resampler_safetensors = os.path.join(model_dir, "resampler.safetensors")
        assert os.path.exists(resampler_safetensors)
        state_dicts = load_file(resampler_safetensors, device='cpu')
        self.resampler.load_state_dict(state_dicts)

        ip_adapter_safetensor = os.path.join(model_dir, "ip_layers.safetensors")
        assert os.path.exists(ip_adapter_safetensor)
        state_dicts = load_file(ip_adapter_safetensor, device='cpu')
        cross_attention_dim = state_dicts['0.to_k_ip.weight'].shape[1]
        if '0.registers' in state_dicts:
            n_registers = state_dicts['0.registers'].shape[1]
        else:
            n_registers = 0
        self.transformer, ip_layers = register_ip_adapter(self.transformer, cross_attention_dim=cross_attention_dim, n_registers=n_registers, dtype=dtype)
        ip_layers.load_state_dict(state_dicts)

    def init_ref_adapter_modules(self, model_dir, dtype=torch.bfloat16):
        ref_adapter_safetensor = os.path.join(model_dir, "ref_layers.safetensors")
        assert os.path.exists(ref_adapter_safetensor)
        state_dicts = load_file(ref_adapter_safetensor, device='cpu')
    
        self.transformer, ref_layers = register_ref_adapter(self.transformer, dtype=dtype)
        ref_layers.load_state_dict(state_dicts)

    def encode_reference_images(self, ref_image_list, ref_prompt="image of a face", drop=False, generator=None):
        # ref_image_list: List[PIL]
        if not isinstance(ref_image_list, list):
            ref_image_list = [ref_image_list,]
        batch_ref_image = torch.stack([torch.tensor(np.array(img)) for img in ref_image_list])
        batch_ref_image = batch_ref_image / 255.0 * 2 - 1  # Normalize to [-1, 1]
        batch_ref_image = batch_ref_image.permute(0, 3, 1, 2)  # BHWC -> BCHW
        batch_ref_image = batch_ref_image[:,:,None] # BCFHW
        if drop:
            batch_ref_image = batch_ref_image * 0

        vae = self.vae
        tokenizer = self.tokenizer
        text_encoder = self.text_encoder
        transformer = self.transformer

        device = vae.device
        weight_dtype = vae.dtype

        batch_ref_image = batch_ref_image.to(device=device, dtype=weight_dtype)
        # batch_ref_image: [B, C, F, H, W]
        vae_feat = vae.encode(batch_ref_image).latent_dist.sample(generator=generator)
        mean, std = model_utils.cal_mean_and_std(vae, vae.device, vae.dtype)
        vae_feat = (vae_feat - mean) * std # [B, C, 1, H, W]
        vae_feat_list = [feat[None] for feat in vae_feat] # List[1, C, 1, H, W]
        ref_prompt_list = [ref_prompt] * len(vae_feat_list)
        ref_text_embeds, _ = model_utils.encode_prompt(
            tokenizer,
            text_encoder,
            ref_prompt_list,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
            dtype=weight_dtype,
        )
        ref_text_embeds_list = [embed[None] for embed in ref_text_embeds]
        ref_buffer = transformer(
            hidden_states=vae_feat_list,
            encoder_hidden_states=ref_text_embeds_list,
            timestep=torch.LongTensor([0]).to(device),
            attention_kwargs={"ref_feature_extractor": True},
            return_dict=False,
        )
        return ref_buffer


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_i: float = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_kwargs_uncond: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The dtype to use for the torch.amp.autocast.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if attention_kwargs_uncond is None:
            attention_kwargs_uncond = attention_kwargs

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs_uncond,
                        return_dict=False,
                    )[0]
                    
                    if guidance_scale_i is not None:
                        noise_i = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + guidance_scale_i * (noise_i - noise_uncond) + guidance_scale * (noise_pred - noise_i)
                    else:
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)