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
import torch
import numpy as np

from safetensors.torch import load_file
from diffusers.callbacks import MultiPipelineCallbacks
from diffusers.utils import logging
from diffusers import WanPipeline
from modules.lite.attention_processor import register_ip_adapter_wan
from modules.common import model_utils_wan as model_utils

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.pipelines.wan.pipeline_wan import *
from modules.lite.trasnformer_wan import WanTransformer3DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LynxLiteWanPipeline(WanPipeline):
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


    def init_image_proj_modules(self, model_dir, device="cuda", dtype=torch.bfloat16):
        from ..common.resampler import Resampler
        self.resampler = Resampler(
            depth=4,
            dim=1280,
            dim_head=64,
            embedding_dim=512,
            ff_mult=4,
            heads=20,
            num_queries=16,
            output_dim=2048,
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

        self.transformer, ip_layers = register_ip_adapter_wan(
            self.transformer, cross_attention_dim=2048, hidden_size=5120, layers=2, dtype=dtype
        )
        ip_layers.load_state_dict(state_dicts)


    def encode_face_embedding(self, face_embeds, do_classifier_free_guidance, device, dtype):
        num_images_per_prompt = 1

        if isinstance(face_embeds, torch.Tensor):
            face_embeds = face_embeds.clone().detach()
        else:
            face_embeds = torch.tensor(face_embeds)

        # TODO: move dim to config
        face_embeds = face_embeds.reshape([1, -1, 512])

        if do_classifier_free_guidance:
            face_embeds = torch.cat([torch.zeros_like(face_embeds), face_embeds], dim=0)
        else:
            face_embeds = torch.cat([face_embeds], dim=0)

        face_embeds = face_embeds.to(device=self.resampler.latents.device, 
                                     dtype=self.resampler.latents.dtype)
        face_embeds = self.resampler(face_embeds)

        bs_embed, seq_len, _ = face_embeds.shape
        face_embeds = face_embeds.repeat(1, num_images_per_prompt, 1)
        face_embeds = face_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        return face_embeds.to(device=device, dtype=dtype)


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        face_embeds: Optional[torch.FloatTensor] = None,
        face_token_embeds: Optional[torch.FloatTensor] = None,
        ip_scale: float = 1.0,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512
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
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
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
            output_type (`str`, *optional*, defaults to `"pil"`):
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

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

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

        if face_token_embeds is None:
            # 3.1 Encoder input face embedding
            face_token_embeds = self.encode_face_embedding(
                face_embeds, self.do_classifier_free_guidance, device=device, dtype=self.transformer.dtype
            )
        else:
            face_token_embeds = torch.cat([torch.zeros_like(face_token_embeds), face_token_embeds], dim=0)

        face_token_embeds = face_token_embeds.to(device=device, dtype=self.transformer.dtype)

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

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            latents = torch.cat([latents] * 2)

        face_token_embeds = face_token_embeds.to(transformer_dtype)

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
                    image_embed=face_token_embeds,
                    ip_scale=ip_scale,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    # uncond_face_token_embeds = uncond_face_token_embeds.to(transformer_dtype)
                    # noise_uncond = self.transformer(
                    #     hidden_states=latent_model_input,
                    #     timestep=timestep,
                    #     image_embed=uncond_face_token_embeds,
                    #     encoder_hidden_states=negative_prompt_embeds,
                    #     attention_kwargs=attention_kwargs,
                    #     return_dict=False,
                    # )[0]
                    noise_uncond, noise_pred = noise_pred.chunk(2)
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