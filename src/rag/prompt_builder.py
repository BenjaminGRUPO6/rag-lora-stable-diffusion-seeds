from __future__ import annotations

from collections.abc import Mapping, Sequence

LABEL_QUERIES = {
    "intact": "calidad de semillas intactas de soja, criterios de aceptación y almacenamiento preventivo",
    "spotted": "manchas visibles en semillas de soja, posibles causas documentadas, prevención y manejo",
    "immature": "semillas de soja inmaduras, causas de desarrollo incompleto y criterios de calidad",
    "broken": "daño mecánico y rotura de semillas de soja, prevención durante cosecha, transporte y almacenamiento",
    "skin_damaged": "daño de cubierta en semillas de soja, posibles causas y medidas de manejo",
}


def build_retrieval_query(label: str, observations: str | None = None) -> str:
    base = LABEL_QUERIES.get(label, "defectos visibles en semillas de soja y control de calidad")
    return f"{base}. Observaciones visuales: {observations.strip()}" if observations else base


def _read_field(fragment: Mapping[str, object], names: Sequence[str]) -> str:
    """Return the first non-empty string value available in a retrieved fragment."""

    for name in names:
        value = fragment.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "No disponible"


def build_report_prompt(
    prediction: dict,
    retrieved: list[dict],
) -> str:
    """Build a prompt for a preliminary report from prediction and RAG evidence."""

    label = str(prediction.get("label", "unknown"))
    confidence = prediction.get("confidence", "unknown")

    evidence_lines: list[str] = []

    for index, fragment in enumerate(retrieved, start=1):
        title = _read_field(fragment, ("title", "document_title", "name"))
        text = _read_field(fragment, ("text", "content", "chunk", "snippet"))
        source = _read_field(fragment, ("source", "url", "path", "citation"))
        evidence_lines.append(
            "\n".join(
                [
                    f"[{index}] Título: {title}",
                    f"Texto: {text}",
                    f"Fuente: {source}",
                ]
            )
        )

    evidence = (
        "\n\n".join(evidence_lines)
        if evidence_lines
        else "No se recuperaron fragmentos técnicos. El informe debe indicarlo como limitación."
    )

    return "\n".join(
        [
            "Genera un informe preliminar en español para una semilla de soja.",
            f"Etiqueta visual estimada: {label}",
            f"Confianza del clasificador: {confidence}",
            "",
            "Evidencia recuperada:",
            evidence,
            "",
            "Instrucciones obligatorias:",
            "- No afirmar un diagnóstico definitivo ni presentar la categoría visual como enfermedad confirmada.",
            "- Incluir citas explícitas a las fuentes recuperadas cuando uses evidencia.",
            "- Incluir limitaciones del análisis, especialmente si no hay evidencia recuperada.",
            "- Tratar `spotted` solo como una categoría visual, no como diagnóstico de hongo.",
            "- Mantener el informe como orientación técnica preliminar.",
        ]
    )
