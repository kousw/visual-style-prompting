import argparse
import os
import random
import sys

import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image

from modules.pipelines.pipeline_sdxl import StableDiffusionXLPipeline


def generate_sdxl(args):
    height = width = args.resolution

    # lcm_lora_id = "latent-consistency/lcm-lora-sdxl"
    # vae_path = "models/sdxl-vae-fp16"

    # Load VAE component
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        # "cagliostrolab/animagine-xl-3.0",
        args.model_name_or_path,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        reference_start_layer_index=args.reference_start_layer_index,
    )

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    # pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.to("cuda")

    if args.reference_image is not None:
        reference_image = Image.open(args.reference_image).convert("RGB")
        reference_image = reference_image.resize((height, width), Image.BILINEAR)
    else:
        reference_image = None

    # pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    for i in range(args.num_samples):
        image = pipeline(
            args.prompt,
            negative_prompt=args.negative_prompt,
            reference_images=reference_image,
            width=width,
            height=height,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            reference_start_layer_index=args.reference_start_layer_index,
            clip_skip=2,
        ).images[0]

        image.save(os.path.join(args.output_dir, f"out_sdxl_{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Model name or path",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--prompt", nargs="*", type=str, default=None, help="Prompt text"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name",
        help="Negative prompt text",
    )
    parser.add_argument(
        "--reference_image", type=str, default=None, help="Reference image path"
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.0, help="Guidance scale"
    )
    parser.add_argument(
        "--reference_start_layer_index",
        type=int,
        default=23,
        help="Reference start layer index(defulth:23(24th layer))",
    )
    parser.add_argument("--resolution", type=int, default=768, help="Resolution")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--disable_feature_map", action="store_true", help="disable feature map"
    )

    args = parser.parse_args()

    if args.seed is None:
        # from int min to max
        args.seed = random.randint(0, sys.maxsize)

    args.batch_size = len(args.prompt)

    generate_sdxl(args)
