from src.rag.prompt_builder import build_report_prompt


def test_prompt_requires_non_definitive_language() -> None:
    prompt = build_report_prompt(
        {"label": "biological_damage", "confidence": 0.81},
        [{"title": "Manual", "text": "Información de prevención."}],
    )
    assert "No afirmar un diagnóstico definitivo" in prompt
    assert "Manual" in prompt
