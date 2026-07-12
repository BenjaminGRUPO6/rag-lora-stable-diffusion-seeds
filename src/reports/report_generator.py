from __future__ import annotations

from src.reports.source_formatter import format_sources


def generate_template_report(prediction: dict, retrieved: list[dict]) -> dict:
    return {
        "result": prediction.get("label", "unknown"),
        "confidence": prediction.get("confidence", 0.0),
        "possible_causes": ["Pendiente de generación basada en fuentes recuperadas."],
        "prevention_and_management": ["Revisar las fuentes y validar con un especialista."],
        "sources": format_sources(retrieved),
        "limitations": (
            "La clasificación se basa en una imagen y no confirma una enfermedad. "
            "Puede requerir inspección adicional o análisis de laboratorio."
        ),
    }
