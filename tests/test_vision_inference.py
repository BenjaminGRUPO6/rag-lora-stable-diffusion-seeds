from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.vision.inference import (
    VisionInferenceEngine,
    build_inference_transform,
    load_resnet18_checkpoint,
    predict_image,
)
from src.pipelines.analyze_seed import analyze_seed


class TinyClassifier(torch.nn.Module):
    """Small classifier used to test checkpoint loading without ResNet weights."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.arange(float(num_classes)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.bias.repeat(int(inputs.shape[0]), 1)


def test_load_resnet18_checkpoint_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_resnet18_checkpoint(tmp_path / "missing.pt", device=torch.device("cpu"), config={})


def test_load_resnet18_checkpoint_available_and_predicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = ["intact", "spotted", "immature", "broken", "skin_damaged"]
    model = TinyClassifier(num_classes=len(labels))
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "class_to_idx": {label: index for index, label in enumerate(labels)},
            "model_state_dict": model.state_dict(),
            "config": {"model": {"architecture": "resnet18", "dropout": 0.2}},
        },
        checkpoint_path,
    )

    def tiny_factory(
        architecture: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
    ) -> TinyClassifier:
        assert architecture == "resnet18"
        assert pretrained is False
        assert dropout == 0.2
        return TinyClassifier(num_classes=num_classes)

    monkeypatch.setattr("src.vision.inference_engine.create_model", tiny_factory)

    loaded_model, loaded_labels, checkpoint = load_resnet18_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
        config={},
    )
    prediction = predict_image(
        model=loaded_model,
        image_path=Image.new("RGB", (16, 16), color="white"),
        transform=build_inference_transform(image_size=16),
        labels=loaded_labels,
        device=torch.device("cpu"),
    )

    assert checkpoint["class_to_idx"]["intact"] == 0
    assert loaded_labels == labels
    assert prediction["label"] == "skin_damaged"
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
    assert prediction["logits"]["skin_damaged"] == pytest.approx(4.0)
    assert [item["label"] for item in prediction["top_3"]] == [
        "skin_damaged",
        "broken",
        "immature",
    ]


def test_inference_engine_matches_analyze_seed_pipeline(tmp_path: Path) -> None:
    labels = ["intact", "spotted", "immature", "broken", "skin_damaged"]
    engine = VisionInferenceEngine(
        model=TinyClassifier(num_classes=len(labels)),
        labels=labels,
        transform=build_inference_transform(image_size=16),
        device=torch.device("cpu"),
    )
    image = Image.new("RGB", (16, 16), color="white")
    vision_config = tmp_path / "vision.yaml"
    rag_config = tmp_path / "rag.yaml"
    vision_config.write_text(
        "\n".join(
            [
                "data:",
                "  image_size: 16",
                "inference:",
                "  confidence_threshold: 0.60",
                "  margin_threshold: 0.15",
            ]
        ),
        encoding="utf-8",
    )
    rag_config.write_text("rag:\n  top_k: 1\n", encoding="utf-8")

    direct = engine.predict_dict(image)
    pipeline = analyze_seed(
        image=image,
        vision_config_path=vision_config,
        rag_config_path=rag_config,
        inference_engine=engine,
        retriever=lambda query, top_k=None: [],
        device_name="cpu",
    )

    assert pipeline["prediction"] == direct["label"]
    assert pipeline["confidence"] == pytest.approx(direct["confidence"])
    assert pipeline["probabilities"] == pytest.approx(direct["probabilities"])
    assert pipeline["logits"] == pytest.approx(direct["logits"])
    assert pipeline["top_3"] == direct["top_3"]
