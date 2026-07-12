from __future__ import annotations

from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def load_pipeline(base_model: str, lora_path: str | Path):
    if not torch.cuda.is_available():
        raise RuntimeError("La generación con SD 1.5 + LoRA requiere una GPU CUDA en esta configuración.")
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16)
    pipe.load_lora_weights(str(lora_path))
    return pipe.to("cuda")


def generate(pipe, prompt: str, output_path: str | Path, seed: int = 42) -> Path:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, generator=generator).images[0]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output
