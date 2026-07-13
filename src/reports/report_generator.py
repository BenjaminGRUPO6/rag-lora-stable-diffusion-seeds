from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from src.reports.source_formatter import format_sources


class ReportProvider(Protocol):
    """Optional generative provider interface for report creation."""

    def generate(
        self,
        prompt: str,
        prediction: dict,
        retrieved_sources: list[dict],
        limitations: list[str],
    ) -> dict | str:
        """Generate a preliminary report from prompt and retrieved evidence."""


def generate_preliminary_report(
    prediction: dict,
    retrieved_sources: list[dict],
    observations: list[str] | None = None,
    uncertainty_status: str = "certain",
    provider: ReportProvider | None = None,
    prompt: str = "",
    limitations: list[str] | None = None,
) -> dict:
    """Generate a preliminary report, falling back to deterministic retrieved fragments."""
    active_limitations = limitations or default_limitations(
        bool(retrieved_sources),
        prediction.get("label", "unknown"),
        uncertainty_status,
    )
    if provider is not None:
        generated = provider.generate(
            prompt=prompt,
            prediction=prediction,
            retrieved_sources=retrieved_sources,
            limitations=active_limitations,
        )
        if isinstance(generated, dict):
            return generated
        return {
            "categoria_estimada": prediction.get("label", "unknown"),
            "confianza": prediction.get("confidence", 0.0),
            "informe_generado": generated,
            "referencias": format_sources(retrieved_sources),
            "limitaciones": active_limitations,
        }

    return generate_deterministic_report(
        prediction=prediction,
        retrieved_sources=retrieved_sources,
        observations=observations,
        uncertainty_status=uncertainty_status,
        limitations=active_limitations,
    )


def generate_deterministic_report(
    prediction: dict,
    retrieved_sources: list[dict],
    observations: list[str] | None = None,
    uncertainty_status: str = "certain",
    limitations: list[str] | None = None,
) -> dict:
    """Build a deterministic report from retrieved fragments only."""
    label = str(prediction.get("label", "unknown"))
    confidence = round(float(prediction.get("confidence", 0.0)), 4)
    active_limitations = limitations or default_limitations(
        bool(retrieved_sources),
        label,
        uncertainty_status,
    )
    evidence = [format_evidence_item(item, index) for index, item in enumerate(retrieved_sources, start=1)]
    management = filter_evidence(
        evidence,
        ("preven", "manejo", "storage", "almacen", "conserv", "handling", "control"),
    )

    return {
        "categoria_estimada": label,
        "confianza": confidence,
        "estado_incertidumbre": uncertainty_status,
        "observaciones": observations or [],
        "resumen_visual": (
            f"Clasificacion visual preliminar: {label} con confianza {confidence:.4f}."
        ),
        "informacion_documental": evidence,
        "posibles_factores_descritos_por_las_fuentes": evidence,
        "prevencion_o_manejo": management,
        "referencias": format_sources(retrieved_sources),
        "limitaciones": active_limitations,
    }


def default_limitations(
    has_evidence: bool,
    label: object,
    uncertainty_status: str,
) -> list[str]:
    """Return standard limitations for the preliminary analysis."""
    limitations = [
        "La salida es una clasificacion visual preliminar y no constituye diagnostico.",
        "No sustituye una evaluacion especializada ni analisis de laboratorio cuando corresponda.",
        "El contenido documental se limita a los fragmentos recuperados por el indice RAG.",
    ]
    if str(label) == "spotted":
        limitations.append(
            "La categoria spotted describe alteraciones visibles; no confirma hongo ni enfermedad."
        )
    if not has_evidence:
        limitations.append("No se recuperaron fuentes, por lo que no se agregan factores documentales.")
    if uncertainty_status == "uncertain":
        limitations.append("La prediccion fue marcada como incierta por confianza baja o margen estrecho.")
    return limitations


def format_evidence_item(item: Mapping[str, object], index: int) -> dict:
    """Return a compact structured evidence item."""
    text = compact_text(str(item.get("text") or item.get("content") or item.get("snippet") or ""))
    return {
        "id": index,
        "title": str(item.get("title") or item.get("document_title") or "Sin titulo"),
        "document_id": str(item.get("document_id") or ""),
        "page": item.get("page"),
        "source": str(item.get("source_url") or item.get("source") or item.get("local_path") or ""),
        "score": item.get("score"),
        "fragment": text,
    }


def compact_text(text: str, limit: int = 650) -> str:
    """Normalize whitespace and truncate long retrieved fragments."""
    compacted = " ".join(text.split())
    if len(compacted) <= limit:
        return compacted
    return f"{compacted[: limit - 3].rstrip()}..."


def filter_evidence(evidence: list[dict], terms: tuple[str, ...]) -> list[dict] | list[str]:
    """Return evidence items containing any requested term, or an abstention."""
    filtered = [
        item
        for item in evidence
        if any(term.lower() in str(item.get("fragment", "")).lower() for term in terms)
    ]
    if filtered:
        return filtered
    if evidence:
        return [
            "No se recuperaron fragmentos especificos de prevencion o manejo dentro del top_k solicitado."
        ]
    return ["No hay evidencia recuperada para sustentar recomendaciones."]


def generate_template_report(prediction: dict, retrieved: list[dict]) -> dict:
    """Backward-compatible wrapper for the deterministic preliminary report."""
    return generate_deterministic_report(
        prediction=prediction,
        retrieved_sources=retrieved,
        observations=None,
        uncertainty_status="certain",
    )
