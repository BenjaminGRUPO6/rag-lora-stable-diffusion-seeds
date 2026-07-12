from pathlib import Path

from PIL import Image

from src.data.audit import audit_dataset, summarize_images
from scripts.audit_dataset import main as audit_cli_main


def test_audit_detects_valid_and_corrupted(tmp_path: Path) -> None:
    category = tmp_path / "intact"
    category.mkdir()
    Image.new("RGB", (300, 300), "white").save(category / "valid.png")
    (category / "broken.png").write_bytes(b"not an image")

    images, corrupted = audit_dataset(tmp_path)
    summary = summarize_images(images, corrupted)

    assert summary["total_valid"] == 1
    assert summary["total_corrupted"] == 1
    assert images.iloc[0]["category"] == "intact"


def test_audit_cli_writes_expected_reports(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    category = dataset / "intact"
    output = tmp_path / "audit_output"
    category.mkdir(parents=True)

    first_image = category / "valid_a.png"
    duplicate_image = category / "valid_a_copy.png"
    Image.new("RGB", (300, 300), "white").save(first_image)
    duplicate_image.write_bytes(first_image.read_bytes())
    Image.new("RGB", (320, 320), "black").save(category / "valid_b.png")
    (category / "corrupted.png").write_bytes(b"not an image")

    exit_code = audit_cli_main(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--min-size",
            "200",
            "--near-duplicate-distance",
            "5",
        ]
    )

    assert exit_code == 0

    expected_files = {
        "summary.json",
        "images.csv",
        "category_distribution.csv",
        "corrupted_files.csv",
        "exact_duplicates.csv",
        "possible_near_duplicates.csv",
    }
    assert expected_files == {path.name for path in output.iterdir() if path.is_file()}
