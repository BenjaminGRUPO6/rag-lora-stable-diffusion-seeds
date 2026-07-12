from __future__ import annotations

import subprocess
from pathlib import Path


def build_command(
    diffusers_repo: str | Path,
    train_data_dir: str | Path,
    output_dir: str | Path,
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    max_train_steps: int = 800,
    validation_prompt: str = (
        "photo of soybeanseed soybean seed with visible defect, "
        "official categories intact spotted immature broken skin_damaged"
    ),
) -> list[str]:
    script = Path(diffusers_repo) / "examples" / "text_to_image" / "train_text_to_image_lora.py"
    if not script.exists():
        raise FileNotFoundError(f"No se encontró el script oficial de Diffusers: {script}")
    return [
        "accelerate",
        "launch",
        str(script),
        f"--pretrained_model_name_or_path={model_id}",
        f"--train_data_dir={Path(train_data_dir)}",
        "--caption_column=text",
        "--resolution=512",
        "--random_flip",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        f"--max_train_steps={max_train_steps}",
        "--learning_rate=1e-4",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--mixed_precision=fp16",
        "--rank=8",
        "--checkpointing_steps=200",
        f"--validation_prompt={validation_prompt}",
        "--num_validation_images=4",
        f"--output_dir={Path(output_dir)}",
    ]


def run_training(command: list[str]) -> None:
    subprocess.run(command, check=True)
