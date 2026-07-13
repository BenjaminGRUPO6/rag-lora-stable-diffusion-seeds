from __future__ import annotations

from collections.abc import Mapping, Sequence


LABEL_QUERIES = {
    "intact": "calidad y conservacion de semillas de soja intactas",
    "broken": "dano mecanico y rotura de semillas de soja, prevencion durante cosecha transporte y almacenamiento",
    "immature": "madurez de semillas de soja, desarrollo incompleto y criterios de calidad",
    "spotted": "alteraciones visibles y manchas en semillas de soja, posibles factores documentados sin diagnostico",
    "skin_damaged": "dano de cubierta en semillas de soja, posibles causas y medidas de manejo",
}


def build_retrieval_query(label: str, observations: str | None = None) -> str:
    """Build the category-specific RAG query for retrieved technical evidence."""
    base = LABEL_QUERIES.get(label, "defectos visibles en semillas de soja y control de calidad")
    suffix = "Recuperar evidencia sobre factores descritos, prevencion, manejo y limitaciones."
    query = f"{base}. {suffix}"
    return f"{query} Observaciones visuales: {observations.strip()}" if observations else query


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
    """Build a prompt for an optional provider from prediction and RAG evidence."""
    label = str(prediction.get("label", "unknown"))
    confidence = prediction.get("confidence", "unknown")
    evidence_lines: list[str] = []

    for index, fragment in enumerate(retrieved, start=1):
        title = _read_field(fragment, ("title", "document_title", "name"))
        text = _read_field(fragment, ("text", "content", "chunk", "snippet"))
        source = _read_field(fragment, ("source_url", "source", "url", "path", "local_path"))
        page = _read_field(fragment, ("page",))
        evidence_lines.append(
            "\n".join(
                [
                    f"[{index}] Titulo: {title}",
                    f"Pagina: {page}",
                    f"Texto: {text}",
                    f"Fuente: {source}",
                ]
            )
        )

    evidence = (
        "\n\n".join(evidence_lines)
        if evidence_lines
        else "No se recuperaron fragmentos tecnicos. El informe debe indicarlo como limitacion."
    )

    return "\n".join(
        [
            "Genera un informe preliminar en espanol para una semilla de soja.",
            f"Etiqueta visual estimada: {label}",
            f"Confianza del clasificador: {confidence}",
            "",
            "Evidencia recuperada:",
            evidence,
            "",
            "Instrucciones obligatorias:",
            "- No afirmar diagnostico definitivo ni presentar la categoria visual como enfermedad confirmada.",
            "- Usar solo la evidencia recuperada; no inventar fuentes ni contenido.",
            "- Incluir referencias explicitas a las fuentes recuperadas cuando uses evidencia.",
            "- Incluir limitaciones del analisis, especialmente si no hay evidencia recuperada.",
            "- Tratar spotted solo como categoria visual; no diagnosticar hongo ni enfermedad.",
            "- Mantener el informe como orientacion tecnica preliminar.",
        ]
    )
