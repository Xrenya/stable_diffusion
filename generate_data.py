import os
import gc
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import argparse
import itertools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np
from tqdm.auto import tqdm

import accelerate
import diffusers
import peft
import transformers

extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]

logger = logging.getLogger(__name__)    
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()

logger.info(f"PyTorch: {torch.__version__}")
logger.info(f"Diffusers: {diffusers.__version__}")
logger.info(f"Transformers: {transformers.__version__}")
logger.info(f"PEFT: {peft.__version__}")
logger.info(f"Accelerate: {accelerate.__version__}")
logger.info(f"CUDA available:{torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("Select Runtime > Change runtime type > GPU in Colab, then rerun.")
logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


def generate_class_images(
    pipe,
    class_prompt: str,
    output_dir: str,
    num_images: int = 10,
    batch_size: int = 2,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generator = itertools.chain(*(output.glob(f"class_{ext}") for ext in extensions))
    already_present = sum(1 for _ in generator)
    remaining = max(0, num_images - already_present)
    if remaining == 0:
        logger.info(f"Found {already_present} prior images; generation is complete.")
        return
    start = already_present
    for offset in tqdm(range(0, remaining, batch_size), desc="Image generation"):
        cur_batch = min(batch_size, remaining - offset)
        images = pipe(
            [class_prompt] * cur_batch,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images
        for index, image in enumerate(images):
            image.save(output / f"class_{start + offset + index:04d}.png")
    logger.info(f"Generation is finished!")
    return


def generate_data(
    model_id: str,
    class_prompt: str,
    class_output: str,
    num_images: int,
    batch_size: int
):
    generator = itertools.chain(*(Path(class_output).glob(f"class_{ext}") for ext in extensions))
    already_present = sum(1 for _ in generator)
    if already_present >= num_images:
        print(
            f"Class data already complete: {already_present} images in "
            f"{class_output}"
        )
    else:
        class_pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id,
            torch_dtype=torch.float16,
            local_files_only=True,
            token=os.environ.get("HF_TOKEN", None)
        ).to("cuda")
        class_pipe.set_progress_bar_config(disable=True)
        class_pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
        generate_class_images(
            pipe=class_pipe,
            class_prompt=class_prompt,
            output_dir=class_output,
            num_images=num_images,
            batch_size=batch_size,
        )
        del class_pipe
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Generate a data with baseline model")
    
    parser.add_argument("--model_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--class_prompt", type=str, default="anime kusanagi")
    parser.add_argument("--class_output", type=str, default="class_output")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)

    args = parser.parse_args()
    
    generate_data(
        args.model_id,
        args.class_prompt,
        args.class_output,
        args.num_images,
        args.batch_size,
    )

if __name__ == "__main__":
    main()