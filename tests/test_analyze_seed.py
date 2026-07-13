from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch
from PIL import Image

from src.pipelines.analyze_seed import analyze_seed


class FakeModel(torch.nn.Module):
    """Model stub that returns close top-two logits for uncertainty tests."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = int(inputs.shape[0])
        logits = torch.tensor([[0.0, 4.0, 3.95, 0.0, 0.0]], dtype=torch.float32)
        return logits.repeat(batch_size, 1)


def _write_configs(tmp_path: Path) -> tuple[Path, Path]:
    vision_config = tmp_path / "vision.yaml"
    rag_config = tmp_path / "rag.yaml"
    vision_config.write_text(
        "\n".join(
            [
                "data:",
                "  image_size: 224",
                "output:",
                "  model_dir: models/vision",
                "inference:",
                "  confidence_threshold: 0.60",
                "  margin_threshold: 0.15",
            ]
        ),
        encoding="utf-8",
    )
    rag_config.write_text(
        "\n".join(
            [
                "rag:",
                "  top_k: 2",
                "  embedding_model: test-model",
                "  normalize_embeddings: true",
            ]
        ),
        encoding="utf-8",
    )
    return vision_config, rag_config


def test_analyze_seed_runs_with_fake_model_and_retriever(tmp_path: Path) -> None:
    vision_config, rag_config = _write_configs(tmp_path)
    image = Image.new("RGB", (32, 32), color="white")
    retrieved = [
        {
            "document_id": "DOC002",
            "title": "Soybean standards",
            "text": "Broken soybeans can be associated with mechanical damage during handling.",
            "local_path": "data/documents/accepted/standards.pdf",
            "page": 2,
            "score": 0.9,
        }
    ]

    def fake_retriever(query: str, top_k: int | None = None) -> list[dict]:
        assert "dano mecanico" in query
        assert "fractura visible" in query
        assert top_k == 2
        return retrieved

    result = analyze_seed(
        image=image,
        vision_config_path=vision_config,
        rag_config_path=rag_config,
        observations=["fractura visible"],
        model=FakeModel(),
        transform=lambda _: torch.zeros(3, 224, 224),
        labels=["intact", "broken", "immature", "spotted", "skin_damaged"],
        retriever=fake_retriever,
        device_name="cpu",
    )

    assert result["prediction"] == "broken"
    assert result["uncertainty_status"] == "uncertain"
    assert result["retrieved_sources"] == retrieved
    assert result["preliminary_report"]["categoria_estimada"] == "broken"
    assert result["preliminary_report"]["informacion_documental"][0]["document_id"] == "DOC002"
    assert "diagnostico" in " ".join(result["limitations"])
    assert set(result["processing_times"]) == {
        "vision_seconds",
        "retrieval_seconds",
        "report_seconds",
        "total_seconds",
    }


def test_analyze_seed_spotted_limitations_do_not_diagnose(tmp_path: Path) -> None:
    vision_config, rag_config = _write_configs(tmp_path)

    result = analyze_seed(
        prediction={
            "label": "spotted",
            "confidence": 0.91,
            "probabilities": {"spotted": 0.91, "intact": 0.03, "broken": 0.02},
        },
        vision_config_path=vision_config,
        rag_config_path=rag_config,
        observations="manchas oscuras",
        retriever=lambda query, top_k=None: [],
    )

    limitations = " ".join(result["limitations"]).lower()
    assert result["prediction"] == "spotted"
    assert result["preliminary_report"]["informacion_documental"] == []
    assert "no confirma hongo" in limitations
    assert "alteraciones visibles" in result["retrieval_query"]


def test_analyze_seed_uses_metadata_fallback_when_embeddings_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RAG should remain local and honest when embedding retrieval is unavailable."""
    vision_config, rag_config = _write_configs(tmp_path)
    index_dir = tmp_path / "vector_db"
    index_dir.mkdir()
    (index_dir / "metadata.json").write_text(
        json.dumps(
            [
                {
                    "chunk_id": "DOC001-p001-c0001",
                    "document_id": "DOC001",
                    "title": "Soybean standards",
                    "topic": "broken/mechanical damage; dano mecanico; rotura",
                    "text": "La rotura de semillas de soja puede relacionarse con dano mecanico durante manejo.",
                    "source_url": "https://example.org/soybean.pdf",
                    "page": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    def fail_faiss(*args: object, **kwargs: object) -> object:
        raise RuntimeError("embedding model unavailable")

    monkeypatch.setattr("src.pipelines.analyze_seed.build_faiss_retriever", fail_faiss)

    result = analyze_seed(
        prediction={
            "label": "broken",
            "confidence": 0.9,
            "probabilities": {"broken": 0.9, "intact": 0.1},
        },
        vision_config_path=vision_config,
        rag_config_path=rag_config,
        index_dir=index_dir,
    )

    assert result["rag_status"] == "metadata_fallback"
    assert result["retrieved_sources"][0]["document_id"] == "DOC001"
    assert "recuperacion lexical local" in " ".join(result["limitations"])


def test_analyze_seed_handles_missing_rag_index_honestly(tmp_path: Path) -> None:
    vision_config, rag_config = _write_configs(tmp_path)

    result = analyze_seed(
        prediction={
            "label": "intact",
            "confidence": 0.82,
            "probabilities": {"intact": 0.82, "broken": 0.18},
        },
        vision_config_path=vision_config,
        rag_config_path=rag_config,
        index_dir=tmp_path / "missing_vector_db",
    )

    assert result["rag_status"] == "unavailable"
    assert result["retrieved_sources"] == []
    assert "RAG no disponible localmente" in " ".join(result["limitations"])
