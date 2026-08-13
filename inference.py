from dataclasses import dataclass, field
from typing import Optional, List

import torch

from generate_data import generate_class_images
from diffusers import StableDiffusionPipeline


@dataclass
class DreamBoothLoRAConfig:
    pretrained_model: str = "Xrenya/TheGhostInTheShell"
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
    num_train_steps=20,
    train_text_encoder=False,
)


def load_lora(base_model: str, adapter_dir: str, prompt: str = ""):
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.load_lora_weights(adapter_dir)
    return pipe

pipe = load_lora(character_config.pretrained_model, character_config.pretrained_model)
pipe.to(device="cuda", dtype=torch.bfloat16)
pipe.unet.eval()
pipe.vae.eval()
pipe.text_encoder.eval()
pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
generate_class_images(
    pipe,
    character_config.instance_prompt,
    character_config.output_dir + "test",
    num_images=30,
    batch_size=2,
)
pipe.to("cpu")
torch.cuda.empty_cache()