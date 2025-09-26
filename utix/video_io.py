# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 

""" This script provides utility functions for video I/O """

from typing import List, Union, Tuple, Optional

import math
import imageio
import numpy as np

from PIL import Image
from .logger import Logger

logger = Logger(__name__)


def save_numpy_to_webp(image_sequence, output_path, fps=24, loop=0):
    """
    Convert numpy array to a webp animation.

    Args:
        image_sequence (`np.ndarray` or a list):
            A list of `np.ndarray` or `np.ndarray` represents the video sequence.
        output_path (`str`): 
            Output WebP file path, should ends with ".webp".
        fps (`int`, default: 24): 
            Number of frames per second, `fps = 1000 / duration`.
        loop (`int`, default: 0): 
            Number of times the animation should loop. 0 means infinite.

    Examples:
    >>> video_generated = pipe(
            prompt=prompt,
            ...
        ).frames[0]
    >>> save_numpy_to_webp(video_generated, output_path, fps=24)
    """

    assert output_path.endswith(".webp"), f"Invalid extension: {output_path}"

    assert isinstance(image_sequence, list) or \
          (isinstance(image_sequence, np.ndarray) and len(image_sequence.shape) == 4), \
          f"image_sequence should be a list or 4-dim array"

    try:
        if not isinstance(image_sequence[0], Image.Image):
            pil_images = [Image.fromarray(im) for im in image_sequence]
        else:
            pil_images = image_sequence

        pil_images[0].save(
            output_path,
            format="WEBP",
            save_all=True,
            append_images=pil_images[1:],
            duration=1000 / fps,
            loop=loop
        )

    except Exception:
        logger.error(f"Fail to export data to webp!")
        raise 


def save_numpy_to_mp4(image_sequence, output_path, fps=24):
    """
    Convert numpy array to a mp4 file.

    Args:
        image_sequence (`np.ndarray` or a list): 
            Input video data with shape [F, H, W, C] of a list of [H, W, C].
        output_path (`str`): 
            Output mp4 file path, should ends with ".mp4".
        fps (`int`, default: 24): 
            Frames per second for the output video.

    Examples:
    >>> video_generated = pipe(
            prompt=prompt,
            ...
        ).frames[0]
    >>> save_numpy_to_mp4(video_generated, output_path, fps=24)
    """

    assert output_path.endswith(".mp4"), f"Invalid extension: {output_path}"

    assert isinstance(image_sequence, list) or \
          (isinstance(image_sequence, np.ndarray) and len(image_sequence.shape) == 4), \
          f"image_sequence should be a list or 4-dim array"

    try:
        if isinstance(image_sequence, list):
            image_sequence = np.stack(image_sequence, axis=0)

        # Extract dimensions
        f, h, w, c = image_sequence.shape

        # Use imageio for mp4 exporting
        with imageio.get_writer(output_path, fps=fps) as writer:
            for i in range(f):
                frame = image_sequence[i].astype(np.uint8)
                writer.append_data(frame)

    except Exception:
        logger.error(f"Fail to export data to mp4!")
        raise 


def save_tensor_to_webp(tensor, output_path, fps=24, loop=0):
    """
    Convert a torch tensor to a WebP animation.

    Args:
        tensor (`torch.tensor`, floating, `[-1., 1.]`, `[F, C, H, W]`):
            A tensor represents the video sequence.
        output_path (`str`): 
            Output WebP file path, should ends with ".webp".
        fps (`int`, default: 24): 
            Number of frames per second, `fps = 1000 / duration`.
        loop (`int`, default: 0): 
            Number of times the animation should loop. 0 means infinite.
    """

    video = ((tensor.permute(0, 2, 3, 1).detach().cpu().numpy() + 1) \
             * 0.5 * 255).astype(np.uint8)

    save_numpy_to_webp(video, output_path, fps=fps, loop=loop)


def save_tensor_to_mp4(tensor, output_path, fps=24):
    """
    Convert a torch tensor to a mp4 file.

    Args:
        tensor (`torch.tensor`, floating, `[-1., 1.]`, `[F, C, H, W]`):
            A tensor represents the video sequence.
        output_path (`str`): 
            Output MP4 file path, should ends with ".mp4".
        fps (`int`, default: 24): 
            Number of frames per second, `fps = 1000 / duration`.
    """

    video = ((tensor.permute(0, 2, 3, 1).detach().cpu().numpy() + 1) \
             * 0.5 * 255).astype(np.uint8)

    save_numpy_to_mp4(video, output_path, fps=fps)