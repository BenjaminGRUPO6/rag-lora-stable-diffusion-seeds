from pathlib import Path

from src.synthetic_data.train_sd15_lora import build_command


def test_lora_command_uses_official_script(tmp_path: Path) -> None:
    script = tmp_path / "examples" / "text_to_image" / "train_text_to_image_lora.py"
    script.parent.mkdir(parents=True)
    script.write_text("# test", encoding="utf-8")
    command = build_command(tmp_path, "dataset", "output", max_train_steps=10)
    assert "accelerate" in command[0]
    assert any("stable-diffusion-v1-5" in item for item in command)
    assert any("max_train_steps=10" in item for item in command)
