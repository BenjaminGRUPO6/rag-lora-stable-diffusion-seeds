from pathlib import Path


def test_visual_settings_declare_five_soybean_classes() -> None:
    content = Path("configs/vision.yaml").read_text(encoding="utf-8")

    assert content.count("num_classes: 5") == 1

    for class_name in [
        "intact",
        "spotted",
        "immature",
        "broken",
        "skin_damaged",
    ]:
        assert f"- {class_name}" in content
