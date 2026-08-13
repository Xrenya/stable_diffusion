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
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from peft import LoraConfig

from generate_data import generate_class_images

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


@dataclass
class DreamBoothLoRAConfig:
    pretrained_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    output_dir: str = "dreambooth_lora_output"
    resolution: int = 512

    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: [
            "to_q", "to_k", "to_v", "to_out.0",
        ]
    )

    instance_prompt: str = "anime ohwx kusanagi"
    class_prompt: str = "anime kusanagi"
    instance_data_dir: str = "./data/kusanagi"
    class_data_dir: Optional[str] = "./class_output"
    use_prior_preservation: bool = True
    num_class_images: int = 150
    prior_loss_weight: float = 1.0

    num_train_steps: int = 800
    learning_rate: float = 7.5e-5
    te_learning_rate: float = 5e-6
    train_text_encoder: bool = False
    batch_size: int = 1
    gradient_accumulation_step: int = 1

    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
    random_flip: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 50


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank  # internal dim size of the LoRA mat (A, B)
        self.alpha = alpha  # magnitude of the adapter
        self.scaling = alpha / rank
        # increase the rank (r), the sum of the matrix multiplication grows larger
        # to keep it stable across different rank choices

        # frozen
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        # learnable low-rank mat
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # to match before training output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen_output = self.linear(x)
        lora_output = self.lora_B(self.lora_A(self.lora_dropout(x)))
        return frozen_output + self.scaling * lora_output


def inject_lora(
    model: nn.Module,
    target_module: List[str],
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
):
    already_adapter = [
        name for name, _ in model.named_parameters() if "lora_" in name
    ]
    if already_adapter:
        raise RuntimeError("LoRA is already exist!")

    matches = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and any(name.endswith(target) for target in target_module)
    ]

    if not matches:
        raise RuntimeError("No nn.Linear was matched: {target_module}")

    module_dict = dict(model.named_modules())
    lora_layers = {}
    for name, module in matches:
        lora = LoRALinear(
            module.in_features,
            module.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ).to(device=module.weight.device, dtype=module.weight.dtype)

        lora.linear = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        lora.linear.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            lora.linear.bias.data.copy_(module.bias.data)

        parent_name, _, child_name = name.rpartition(".")
        parent = module_dict[parent_name] if parent_name else model
        if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent[int(child_name)] = lora
        else:
            setattr(parent, child_name, lora)
        lora_layers[name] = lora

    model.requires_grad_(False)
    for lora in lora_layers.values():
        lora.lora_A.weight.requires_grad_(True)
        lora.lora_B.weight.requires_grad_(True)

    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.debug(f"Total {len(lora_layers)} layer with ({n_lora,}) parameters")

    return lora_layers


class DreamBoothLoRADataset(Dataset):

    IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(
        self,
        instance_data_dir: str,
        class_data_dir: Optional[str],
        instance_prompt: str,
        class_prompt: str,
        tokenizer,
        size: int = 512,
        use_prior_preservation: bool = True,
        random_flip: bool = False,
    ):
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.tokenizer = tokenizer

        instance_root = Path(instance_data_dir)
        self.instance_images = sorted(
            p for p in instance_root.iterdir() if p.suffix.lower() in self.IMG_EXT
        )
        if not self.instance_images:
            raise ValueError(f"No supported images found in '{instance_root}'")

        self.class_images = []
        if use_prior_preservation:
            if not class_data_dir:
                raise ValueError("'class_data_dir' is required when prior preservationis used")
            class_root = Path(class_data_dir)
            if not class_root.is_dir():
                raise FileNotFoundError(
                    f"Class directory does not exist: {class_root}. Generate class images first."
                )
            self.class_images = sorted([
                p for p in class_root.iterdir() if p.suffix.lower() in self.IMG_EXT
            ])

            if not self.class_images:
                raise ValueError(f"Prior preservation is enabled but {class_root} is empty")

        self.num_instance = len(self.instance_images)
        self.num_class = len(self.class_images)
        self._length = max(self.num_instance, self.num_class or 0)

        augmentations = [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop(size),
        ]
        if random_flip:
            augmentations.append(transforms.RandomHorizontalFlip())

        augmentations.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.transform = transforms.Compose(augmentations)

    def __len__(self):
        return self._length

    def _caption(self, image_path: Path, fallback: str) -> str:
        text = image_path.with_suffix(".txt")
        if text.is_file():
            caption = text.read_text(encoding="utf-8").strip()
            if caption:
                return caption
        return fallback

    def _tokenize(self, prompt: str) -> torch.Tensor:
        return self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids.squeeze(0)

    def _load(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return self.transform(image)

    def __getitem__(self, idx: int):
        instance_path = self.instance_images[idx % self.num_instance]
        data = {
            "instance_pixel_values": self._load(instance_path),
            "instance_input_ids": self._tokenize(
                self._caption(instance_path, self.instance_prompt)
            ),
        }
        if self.num_class:
            class_path = self.class_images[idx % self.num_class]
            data["class_pixel_values"] = self._load(class_path)
            data["class_input_ids"] = self._tokenize(self.class_prompt)
        return data


def compute_loss(
    unet, vae, text_encoder,
    pixel_values, input_ids,
    noise_scheduler,
    weight_dtype,
):
    pixel_values = pixel_values.to(dtype=weight_dtype)
    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        text_embeddings = text_encoder(input_ids, return_dict=False)[0]

    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.size(0),),
        device=latents.device
    ).long()
    noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps)
    model_pred = unet(
        noisy_latent,
        timesteps,
        encoder_hidden_states=text_embeddings,
        return_dict=False,
    )[0]

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(
            f"Unsupported prediction type: {noise_scheduler.config.prediction_type}"
        )
    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


def train_dreambooth_lora(
    config: DreamBoothLoRAConfig,
    unet, vae, text_encoder,
    tokenizer,
):
    if any("lora_" in name for name, _ in unet.named_parameters()):
        raise RuntimeError("LoRA is already exist!")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_step,
        mixed_precision=config.mixed_precision,
    )
    torch.manual_seed(config.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.to(device=accelerator.device, dtype=weight_dtype)
    vae.to(device=accelerator.device, dtype=weight_dtype)
    text_encoder.to(device=accelerator.device, dtype=weight_dtype)

    unet.add_adapter(LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=config.target_modules,
    ))
    if config.train_text_encoder:
        text_encoder.add_adapter(LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        ))

    train_models = [unet] + ([text_encoder] if config.train_text_encoder else [])
    if accelerator.mixed_precision in ("fp16", "bf16"):
        cast_training_params(train_models, dtype=torch.float32)
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if config.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    unet_params = [p for p in unet.parameters() if p.requires_grad]
    text_params = [p for p in text_encoder.parameters() if p.requires_grad]
    named_trainable = [
        name for name, p in unet.named_parameters() if p.requires_grad
    ]
    if not unet_params:
        raise RuntimeError("PEFT attached zero trainable U-Net parameters")
    if not all("lora_" in name for name in named_trainable):
        raise RuntimeError("No trainable LoRA parameter")

    parameter_groups = [{"params": unet_params, "lr": config.learning_rate}]
    if text_params:
        parameter_groups.append(
            [{"params": text_params, "lr": config.te_learning_rate}]
        )
    if config.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(parameter_groups, weight_decay=1e-2)

    dataset = DreamBoothLoRADataset(
        instance_data_dir=config.instance_data_dir,
        class_data_dir=config.class_data_dir,
        instance_prompt=config.instance_prompt,
        class_prompt=config.class_prompt,
        tokenizer=tokenizer,
        size=config.resolution,
        use_prior_preservation=config.use_prior_preservation,
        random_flip=config.random_flip,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        config.pretrained_model, subfolder="scheduler", 
    )
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.num_train_steps,
    )

    if config.train_text_encoder:
        unet, text_encoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, dataloader, lr_scheduler
        )
    else:
        unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, dataloader, lr_scheduler
        )

    n_trainable = sum(p.numel() for p in unet_params + text_params)
    logger.debug(f"Attached one PEFT adapter: {n_trainable:,} trainable parameters")
    logger.debug(f"Trainable U-Net tensors: {len(named_trainable)}")

    if config.lora_rank == 8 and len(config.target_modules) == 4:
        assert n_trainable >= 1_594_368, "Unexpected SD 1.5 rank-8 adapter size"

    unet.train()
    if config.train_text_encoder:
        text_encoder.train()
    else:
        text_encoder.eval()
    vae.eval()

    update_step = 0
    progress = tqdm(
        total=config.num_train_steps, desc="DreamBooth+LoRA training"
    )
    optimizer.zero_grad(set_to_none=True)
    while update_step < config.num_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                instance_loss = compute_loss(
                    unet,
                    vae,
                    text_encoder,
                    batch["instance_pixel_values"].to(accelerator.device),
                    batch["instance_input_ids"].to(accelerator.device),
                    noise_scheduler,
                    weight_dtype,
                )
                prior_loss = compute_loss(
                    unet,
                    vae,
                    text_encoder,
                    batch["class_pixel_values"].to(accelerator.device),
                    batch["class_input_ids"].to(accelerator.device),
                    noise_scheduler,
                    weight_dtype,
                )
                loss = instance_loss + config.prior_loss_weight * prior_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet_params + text_params, config.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                update_step += 1
                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss.detach().item():.4f}",
                    instance=f"{instance_loss.detach().item():.4f}",
                    prior=f"{prior_loss.detach().item():.4f}",
                )
                if update_step >= config.num_train_steps:
                    break
    progress.close()
    accelerator.wait_for_everyone()
    return (
        accelerator.unwrap_model(unet, keep_fp32_wrapper=False),
        accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=False)
        if config.train_text_encoder else text_encoder,
    )


def save_dreambooth_lora(
    unet,
    output_dir: str,
    text_encoder=None,
    train_text_encoder: bool = False,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    unet_state = diffusers.utils.convert_state_dict_to_diffusers(
        peft.utils.get_peft_model_state_dict(unet)
    )
    text_state = None
    if train_text_encoder:
        text_state = diffusers.utils.convert_state_dict_to_diffusers(
            peft.utils.get_peft_model_state_dict(text_encoder)
        )
    diffusers.loaders.StableDiffusionLoraLoaderMixin.save_lora_weights(
        save_directory=output_dir,
        unet_lora_layers=unet_state,
        text_encoder_lora_layers=text_state,
        safe_serialization=True,
    )
    path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    return path


def load_lora(base_model: str, adapter_dir: str, prompt):
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.load_lora_weights(adapter_dir)
    image = pipe(prompt, num_inference_steps=20).images[0]
    print("Adapter reload and inference succeeded.")
    return image


character_config = DreamBoothLoRAConfig(
    instance_data_dir="./data/kusanagi",
    class_data_dir="./class_output",
    output_dir="./output/kusanagi_character",
    instance_prompt="anime ohwx kusanagi",
    class_prompt="anime kusanagi",
    use_prior_preservation=True,
    num_class_images=100,
    lora_rank=16,
    lora_alpha=16,
    learning_rate=7.5e-5,
    num_train_steps=800,
    train_text_encoder=False,
)

pipe = diffusers.StableDiffusionPipeline.from_pretrained(
        character_config.pretrained_model,
        torch_dtype=torch.float16,
    )
pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))

trained_unet, trained_text_encoder = train_dreambooth_lora(
    character_config,
    pipe.unet,
    pipe.vae,
    pipe.text_encoder,
    pipe.tokenizer,
)
save_dreambooth_lora(
    trained_unet,
    character_config.output_dir,
    trained_text_encoder,
    character_config.train_text_encoder,
)

pipe.unet = trained_unet
pipe.text_encoder = trained_text_encoder
pipe.to(device="cuda", dtype=torch.bfloat16)
pipe.unet.eval()
pipe.vae.eval()
pipe.text_encoder.eval()
pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
generate_class_images(
    pipe,
    character_config.instance_prompt,
    character_config.output_dir,
    num_images=20,
    batch_size=2,
)
pipe.to("cpu")
torch.cuda.empty_cache()
