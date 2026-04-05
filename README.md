# Tutorial: Stable Diffusion — Generations at a Glance

Introduction to Stable Diffusion

| Feature | SD 1 (v1.x) | SD 2 (v2.x) | SD 3 (v3.x) |
|---|---|---|---|
| **Architecture** | Latent Diffusion + UNet (~865 M params) + VAE | Same latent-diffusion core with larger UNet (≈860 M–1 B), tweaked attention blocks, optional depth & inpaint heads | Diffusion-Transformer (DiT-like) + Mixture-of-Experts, 800 M → 8 B params |
| **Text encoder** | CLIP ViT-L/14 (OpenAI) | OpenCLIP ViT-G/14 (larger, open) | Triple encoder: CLIP-L + CLIP-G + T5-XXL — better long prompts & text rendering |
| **Latent → pixel** | 64×64 → 512×512 | 96×96 → 768×768 | 128×128 → 1024×1024 |
| **Latent channels** | 4 | 4 | 16 |
| **VAE scale** | 0.18215 | 0.18215 | scale 1.5305, shift 0.0609 |
| **Training data** | LAION-2B subset (~2.3 B images) | LAION-5B aesthetic-filtered; fewer copyrighted terms | Proprietary + licensed stock; designed for accurate text & multi-object composition |
| **New capabilities** | First public T2I model; fast, ~4 GB VRAM | Sharper native images; Depth-to-Image; Inpainting; better anatomy | Much better spelling & logos; complex multi-subject scenes; fewer artifacts |
| **Safety / filters** | Opt-in NSFW filter script | Heavier prompt + image filters; many artist names removed | Strict alignment & watermarking; gated API at launch |

## Content

- Weight-compatible models implemented:
  - [x] **SD1.x**

  <table>
    <tr>
      <td align=”center”><b>txt2img</b></td>
      <td align=”center”><b>Original</b></td>
      <td align=”center”><b>img2img</b></td>
    </tr>
    <tr>
      <td align=”center”><img src=”images/golden_castle.png” width=”256”></td>
      <td align=”center”><img src=”images/original_cat.png” width=”256”></td>
      <td align=”center”><img src=”images/cat_with_black_glasses.png” width=”256”></td>
    </tr>
  </table>

  - [x] **SD2.x**

  <table>
    <tr>
      <td align=”center”><b>txt2img</b></td>
      <td align=”center”><b>Original</b></td>
      <td align=”center”><b>img2img</b></td>
    </tr>
    <tr>
      <td align=”center”><img src=”images/castle.png” width=”256”></td>
      <td align=”center”><img src=”images/original_cat.png” width=”256”></td>
      <td align=”center”><img src=”images/cat_with_glasses.png” width=”256”></td>
    </tr>
  </table>

  - [x] **SD3.x**

- Finetuning:
  - [x] DreamBooth
  - [x] LoRA
  - [x] LoRA + DreamBooth

