# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import os
import torch
import numpy as np

from PIL import Image
from diffusers import UniPCMultistepScheduler

from modules.common.inference_utils import SubjectInfo, VideoStyleInfo, dtype_mapping
from utix import Logger, save_numpy_to_mp4, save_numpy_to_webp
from .lynx_pipeline import LynxWanPipeline

logger = Logger(__name__)


class LynxWanInfer():
    def __init__(
        self,
        adapter_path: str = None,
        base_model_path: str = None,
        pipe: LynxWanPipeline = None,
        device: Union[str, torch.device] = "cuda",
        dtype: Union[str, torch.dtype] = "bf16"
    ) -> None:
        logger.info("Initializing pipeline")
        if adapter_path is not None:
            assert pipe is None, "Model path is already provided!"
            dtype = dtype_mapping[dtype] if isinstance(dtype, str) else dtype

            loaded = LynxWanInfer.load_pipeline_and_models(
                adapter_path, base_model_path, device=device, dtype=dtype
            )
            self.pipe = loaded["pipe"]

        else:
            assert pipe, "Should provide model path or a pipe object!"
            self.pipe = pipe

        assert self.pipe, "Init pipeline failed!"

        logger.info("Initializing NSFW classifier")
        from transformers import pipeline
        self.nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

    def generate_t2v(
        self,
        subject_info: SubjectInfo,
        style_info: VideoStyleInfo,
        output_dir: str,
        ext: str = "mp4",
        fps: int = 16,
        **override_kwargs
    ) -> None:
        logger.info(f"Generating video for style: {style_info.style_name}")

        # Override the style info args
        for k in override_kwargs:
            setattr(style_info, k, override_kwargs[k])

        if hasattr(self.pipe, "resampler"):
            arcface_embed = torch.from_numpy(subject_info.face_embeds)
            arcface_embed = arcface_embed.to(device=self.pipe.device, dtype=self.pipe.dtype)
            arcface_embed = arcface_embed[None,None,:]
            face_embeds = self.pipe.resampler(arcface_embed)
            ip_hidden_states = [face_embeds]
            face_embeds_uncond = self.pipe.resampler(arcface_embed * 0)
            ip_hidden_states_uncond = [face_embeds_uncond]
        else:
            ip_hidden_states = None
            ip_hidden_states_uncond = None

        if hasattr(self.pipe.transformer.blocks[0].attn1.processor, "to_k_ref"):
            from ..common.face_utils import align_face
            aligned_face_image_pil = align_face(subject_info.image_pil, subject_info.landmarks, extend_face_crop=True, face_size=256)
            aligned_face_image_np = np.array(aligned_face_image_pil)
            ref_generator = torch.Generator().manual_seed(style_info.seed + 1) if style_info.seed >= 0 else None
            ref_buffer = self.pipe.encode_reference_images([aligned_face_image_pil], generator=ref_generator)
            ref_generator = torch.Generator().manual_seed(style_info.seed + 1) if style_info.seed >= 0 else None
            ref_buffer_uncond = self.pipe.encode_reference_images([aligned_face_image_pil], drop=True, generator=ref_generator)
        else:
            ref_buffer = None
            ref_buffer_uncond = None
        
        generator = torch.Generator().manual_seed(style_info.seed) if style_info.seed >= 0 else None

        result_frames = self.pipe(
            prompt=style_info.prompt,
            negative_prompt=style_info.negative_prompt,
            height=style_info.height,
            width=style_info.width,
            num_inference_steps=style_info.num_inference_steps,
            num_frames=style_info.num_frames,
            guidance_scale=style_info.guidance_scale,
            guidance_scale_i=getattr(style_info, "guidance_scale_i", None),
            generator=generator, 
            output_type="pil",
            attention_kwargs={"ip_hidden_states": ip_hidden_states, "ip_scale": style_info.ip_scale, "ref_buffer": ref_buffer, "ref_scale": style_info.ref_scale},
            attention_kwargs_uncond={"ip_hidden_states": ip_hidden_states_uncond, "ip_scale": style_info.ip_scale, "ref_buffer": ref_buffer_uncond, "ref_scale": style_info.ref_scale},
        ).frames[0]

        # Safety check
        logger.info("Running first NSFW classifier")
        nsfw_scores = []
        for frame in result_frames:
            for item in self.nsfw_classifier(frame):
                if item['label'] == 'nsfw':
                    nsfw_score = item['score']
            nsfw_scores.append(nsfw_score)
        nsfw_score = max(nsfw_scores)
        
        logger.info("Running second NSFW classifier")
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        import opennsfw2 as n2
        nsfw_scores2 = n2.predict_images(result_frames)
        nsfw_score2 = max(nsfw_scores2)
        
        if nsfw_score >= 0.85 or nsfw_score2 >= 0.75:
            logger.warning("NSFW detected! Not saving video")
        else:
            result_frames = np.array(result_frames)
            out_video_name = "{sub}/{style}-fr{frame}-s{seed}.{ext}".format(
                sub=subject_info.name,
                style=style_info.style_name,
                frame=style_info.num_frames,
                seed=style_info.seed,
                ext=ext
            )

            out_video_path = os.path.join(output_dir, out_video_name)
            os.makedirs(os.path.dirname(out_video_path), exist_ok=True)

            self._export_video(result_frames, out_video_path, fps)

    @staticmethod
    def load_pipeline_and_models(
        adapter_path: str,
        base_model_path: str = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        save_memory: bool = False
    ) -> LynxWanPipeline:

        loaded = {}
        
        pipe = LynxWanPipeline.from_pretrained(base_model_path, torch_dtype=dtype)

        # Use UniPCMultistepScheduler for potentially better quality.
        pipe.scheduler = UniPCMultistepScheduler.from_pretrained(
            base_model_path, subfolder="scheduler", torch_dtype=dtype
        )
        
        if adapter_path is not None:
            logger.info("Loading adapter layers")
            pipe.init_image_proj_modules(adapter_path, device=device, dtype=dtype)
            pipe.init_ref_adapter_modules(adapter_path)

        pipe.to(device)

        if save_memory:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()

        loaded["pipe"] = pipe

        return loaded

    def _export_video(self, frames: np.ndarray, path: str, fps: int = 8) -> None:
        if path.endswith("webp"):
            # Export to webp for easy preview
            save_numpy_to_webp(frames, path, fps=fps)
        else:
            save_numpy_to_mp4(frames, path, fps=fps)
