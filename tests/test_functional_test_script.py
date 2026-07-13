from __future__ import annotations

from scripts.run_functional_test import (
    contains_forbidden_diagnostic,
    sources_real_when_available,
)


def test_contains_forbidden_diagnostic_detects_affirmative_claims() -> None:
    assert contains_forbidden_diagnostic("La enfermedad confirmada requiere manejo.")
    assert not contains_forbidden_diagnostic("No constituye diagnostico.")


def test_sources_real_when_available_requires_metadata() -> None:
    assert sources_real_when_available([])
    assert sources_real_when_available(
        [{"document_id": "DOC001", "title": "Fuente", "text": "Fragmento"}]
    )
    assert not sources_real_when_available([{"document_id": "DOC001", "text": "Fragmento"}])
