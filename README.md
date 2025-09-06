# Stable Diffusion “Generations” at a Glance
Introduction to Stable Diffusion

| Feature / Aspect | Stable Diffusion 1 (v1.x) | Stable Diffusion 2 (v2.x) | Stable Diffusion 3 (v3.x) |
|------------------|---------------------------|---------------------------------|-----------------------|
| Core architecture | Latent Diffusion with a  UNet (~ 865 M params) + VAE | Same latent-diffusion idea but:  • Larger UNet (≈ 860–1 B)  • Tweaked attention blocks  • Optional depth & inpaint heads | Brand-new Diffusion-Transformer stack (DiT-like) + Mixture-of-Experts. Sizes 800 M → 8 B parameters |
| Text encoder | CLIP ViT-L/14 (OpenAI) | OpenCLIP ViT-G/14 (larger) | Multi-token Transformer (rumoured T5/DeepFloyd-style) for better long prompts & text rendering |
| Native latent ⇆ pixel resolution | 64×64 latent → 512×512 pixels | 96×96 latent → 768×768 pixels (can be 512 or 1024 w/ upscaler) | 128×128 latent → 1024×1024 pixels (targets 4K w/ refiner) |
| Training data | LAION-2B subset (≈ 2.3 B captioned images) | LAION-5B aesthetic & caption-filtered subset; fewer copyrighted terms & porn | New proprietary dataset (+ licensed stock) designed for:  • accurate text  • multi-object composition  • fewer copyright issues |
| Notable new capabilities | – First truly public T2I model– Fast, light (4 GB) | + Sharper, bigger native images, Depth-to-Image, Inpainting models, Better hands/legs vs v1 | + Significantly better spelling & logos, Multi-subject prompts (“a cat playing chess with a robot on Mars”) keep identities separate Fewer artifacts, higher global coherence |
| Safety / filters | opt-in NSFW filter script | Heavier prompt + image safety filters; removed many trademark artist names | Even stricter alignment & watermarking; inference only via gated API (for now) |

