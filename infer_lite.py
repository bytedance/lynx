# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 

import os
import argparse

import torch
import numpy as np
from PIL import Image

from modules.common.face_encoder import FaceEncoderArcFace, get_landmarks_from_image
from modules.common.inference_utils import SubjectInfo, VideoStyleInfo, dtype_mapping

from modules.lite.lynx_lite_infer import LynxLiteWanInfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple single-GPU inference for Lynx (Wan + IPA + Ref)"
    )

    # Required-ish (with defaults matching README layout)
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="models/Wan2.1-T2V-14B-Diffusers",
        help="Path to Wan2.1 base model directory",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="models/lynx_lite",
        help="Path to Lynx adapter directory (resampler/ip/ref layers)",
    )

    # Minimal inputs
    parser.add_argument(
        "--subject_image",
        type=str,
        required=True,
        help="Path to the subject face image (single image)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, blurred background, static, subtitles, style, works, paintings, images, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        help="Optional negative prompt",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_lite",
        help="Output directory for generated video",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="mp4",
        choices=["mp4", "webp"],
        help="Output format",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="demo",
        help="Style name used in output filename",
    )

    # Generation parameters (defaults mirror repo examples)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--guidance_scale_i", type=float, default=2.0)
    parser.add_argument("--ip_scale", type=float, default=1.0)

    # Runtime
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model precision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device (single GPU)",
    )

    return parser.parse_args()


def build_subject_info(image_path: str, device: str) -> SubjectInfo:
    image_pil = Image.open(image_path).convert("RGB")

    # Landmarks
    landmarks = get_landmarks_from_image(image_pil)

    # Face embedding via ArcFace
    face_encoder = FaceEncoderArcFace()
    face_encoder.init_encoder_model("cuda" if device.startswith("cuda") else device)
    embeds = face_encoder(image_pil, need_proc=True, landmarks=landmarks)
    embeds = np.array(embeds.squeeze(0).cpu())

    name = os.path.splitext(os.path.basename(image_path))[0]

    return SubjectInfo(
        name=name,
        image_pil=image_pil,
        landmarks=landmarks,
        face_embeds=embeds,
    )


def build_style_info(args: argparse.Namespace) -> VideoStyleInfo:
    if os.path.isfile(args.prompt):
          with open(args.prompt, 'r') as f:
              args.prompt = f.read().strip()

    return VideoStyleInfo(
        style_name=args.name,
        num_frames=args.num_frames,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        width=args.width,
        height=args.height,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
    )


def main():
    args = parse_args()

    # Prepare subject/style
    subject = build_subject_info(args.subject_image, args.device)
    style = build_style_info(args)

    # Init pipeline
    infer = LynxLiteWanInfer(
        adapter_path=args.adapter_path,
        base_model_path=args.base_model_path,
        dtype=dtype_mapping[args.torch_dtype],
        device=args.device,
    )

    # Generate
    infer.generate_t2v(
        subject_info=subject,
        style_info=style,
        output_dir=args.output_dir,
        ext=args.ext,
        fps=args.fps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_i=args.guidance_scale_i,
        ip_scale=args.ip_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
