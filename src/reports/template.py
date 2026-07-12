from __future__ import annotations


def build_report_payload(label: str, confidence: float, evidence: list[dict], observations: str = "") -> dict:
    return {
        "categoria_estimada": label,
        "confianza": round(float(confidence), 4),
        "observaciones": observations,
        "evidencia_recuperada": evidence,
        "limitacion": "Clasificación visual preliminar; no constituye diagnóstico fitosanitario definitivo.",
        "requiere_revision_humana": True,
    }
