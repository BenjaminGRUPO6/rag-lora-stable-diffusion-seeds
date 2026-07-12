from pathlib import Path

from PIL import Image

from src.data.audit import audit_dataset, summarize_images


def test_audit_detects_valid_and_corrupted(tmp_path: Path) -> None:
    category = tmp_path / "healthy"
    category.mkdir()
    Image.new("RGB", (300, 300), "white").save(category / "valid.png")
    (category / "broken.png").write_bytes(b"not an image")

    images, corrupted = audit_dataset(tmp_path)
    summary = summarize_images(images, corrupted)

    assert summary["total_valid"] == 1
    assert summary["total_corrupted"] == 1
    assert images.iloc[0]["category"] == "healthy"
