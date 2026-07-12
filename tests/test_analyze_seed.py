from src.pipelines.analyze_seed import analyze_seed


def test_analyze_seed_imports_and_builds_report_payload() -> None:
    prediction = {"label": "spotted", "confidence": 0.82}
    retrieved = [
        {
            "title": "Guia tecnica de soja",
            "text": "Las manchas visibles requieren revision y evidencia adicional.",
            "source": "documento-tecnico.pdf",
        }
    ]

    def fake_retriever(query: str) -> list[dict]:
        assert "spotted" in query
        return retrieved

    result = analyze_seed(
        prediction=prediction,
        retriever=fake_retriever,
        observations=["manchas oscuras visibles"],
    )

    assert set(result) == {"query", "retrieved", "prompt", "report"}
    assert result["retrieved"] == retrieved
    assert "spotted" in result["query"]
    assert "spotted" in result["prompt"]
    assert "0.82" in result["prompt"]
    assert "No afirmar un diagnóstico definitivo" in result["prompt"]
    assert result["report"]["result"] == "spotted"
