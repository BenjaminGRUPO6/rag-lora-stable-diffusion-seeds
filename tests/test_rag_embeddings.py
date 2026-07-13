from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from src.rag.embeddings import TextEmbedder


def test_text_embedder_uses_local_files_only_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding model loading should not require network access by default."""
    calls: list[dict[str, object]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            calls.append({"model_name": model_name, **kwargs})

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    embedder = TextEmbedder("test-model")

    assert embedder.local_files_only is True
    assert calls == [{"model_name": "test-model", "local_files_only": True}]
