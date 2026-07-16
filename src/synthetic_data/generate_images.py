from __future__ import annotations

from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def load_pipeline(base_model: str, lora_path: str | Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    if device == "cpu":
        print("Atención: Ejecutando en CPU. Esto tomará más tiempo de lo normal.")
        
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.safety_checker = None
    pipe.load_lora_weights(str(lora_path))
    return pipe.to(device)


def generate(
    pipe, 
    prompt: str, 
    output_path: str | Path, 
    negative_prompt: str | None = None,
    steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42
) -> Path:
    device = pipe.device
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt,
        num_inference_steps=steps, 
        guidance_scale=guidance_scale, 
        generator=generator
    ).images[0]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output
