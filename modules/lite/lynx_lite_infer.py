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
from .lynx_lite_pipeline import LynxLiteWanPipeline

logger = Logger(__name__)


class LynxLiteWanInfer():
    def __init__(
        self,
        adapter_path: str = None,
        base_model_path: str = None,
        pipe: LynxLiteWanPipeline = None,
        device: Union[str, torch.device] = "cuda",
        dtype: Union[str, torch.dtype] = "bf16"
    ) -> None:
        logger.info("Initializing pipeline")
        if adapter_path is not None:
            dtype = dtype_mapping[dtype] if isinstance(dtype, str) else dtype

            loaded = LynxLiteWanInfer.load_pipeline_and_models(
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
        fps: int = 24,
        **override_kwargs
    ) -> None:
        logger.info(f"Generating video for style: {style_info.style_name}")

        # Override the style info args
        for k in override_kwargs:
            setattr(style_info, k, override_kwargs[k])
        
        generator = torch.Generator().manual_seed(style_info.seed) if style_info.seed >= 0 else None

        # Generate
        result_frames = self.pipe(
            prompt=style_info.prompt,
            negative_prompt=style_info.negative_prompt,
            face_embeds=subject_info.face_embeds,
            ip_scale=getattr(style_info, "ip_scale", 1.0),
            height=style_info.height,
            width=style_info.width,
            num_inference_steps=style_info.num_inference_steps,
            num_frames=style_info.num_frames,
            guidance_scale=style_info.guidance_scale,
            generator=generator,
            output_type="pil"
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
    ) -> LynxLiteWanPipeline:

        loaded = {}

        pipe = LynxLiteWanPipeline.from_pretrained(base_model_path, torch_dtype=dtype)

        # Use UniPCMultistepScheduler for potentially better quality.
        pipe.scheduler = UniPCMultistepScheduler.from_pretrained(
            base_model_path, subfolder="scheduler", torch_dtype=dtype
        )

        if adapter_path is not None:
            logger.info("Loading adapter layers")
            pipe.init_image_proj_modules(adapter_path, device=device, dtype=dtype)

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