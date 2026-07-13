from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.vision.dataset import EXPECTED_CLASSES, class_distribution, load_datasets


def test_load_datasets_preserves_expected_class_order(tmp_path: Path) -> None:
    """ImageFolder splits must use the configured soybean class order."""
    _make_image_dataset(tmp_path, images_per_class=2)

    datasets = load_datasets(data_root=tmp_path, classes=EXPECTED_CLASSES, image_size=32)

    assert datasets.class_to_idx == {
        "intact": 0,
        "spotted": 1,
        "immature": 2,
        "broken": 3,
        "skin_damaged": 4,
    }
    assert datasets.train.class_to_idx == datasets.validation.class_to_idx
    assert datasets.validation.class_to_idx == datasets.test.class_to_idx
    assert class_distribution(datasets.train, datasets.class_to_idx) == {
        class_name: 2 for class_name in EXPECTED_CLASSES
    }


def test_load_datasets_rejects_unexpected_classes(tmp_path: Path) -> None:
    """Unexpected split folders should fail before training starts."""
    _make_image_dataset(tmp_path, images_per_class=1)
    (tmp_path / "train" / "unexpected").mkdir()

    try:
        load_datasets(data_root=tmp_path, classes=EXPECTED_CLASSES, image_size=32)
    except ValueError as exc:
        assert "extra" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unexpected class folder.")


def _make_image_dataset(root: Path, images_per_class: int) -> None:
    for split in ("train", "validation", "test"):
        for class_index, class_name in enumerate(EXPECTED_CLASSES):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(images_per_class):
                image = Image.new(
                    "RGB",
                    (40, 40),
                    color=(class_index * 30, image_index * 20, 120),
                )
                image.save(class_dir / f"{class_name}_{image_index}.jpg")
