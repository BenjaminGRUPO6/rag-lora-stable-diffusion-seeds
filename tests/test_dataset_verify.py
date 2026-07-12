from pathlib import Path

from src.data.verify import EXPECTED_CLASSES, verify_dataset


def test_verify_complete_structure(tmp_path: Path) -> None:
    for class_name in EXPECTED_CLASSES:
        folder = tmp_path / class_name
        folder.mkdir()
        (folder / "sample.jpg").write_bytes(b"not-opened-by-verifier")
    result = verify_dataset(tmp_path)
    assert result.valid
    assert result.total == len(EXPECTED_CLASSES)


def test_verify_missing_class(tmp_path: Path) -> None:
    (tmp_path / "intact").mkdir()
    result = verify_dataset(tmp_path)
    assert not result.valid
    assert "spotted" in result.missing_classes
