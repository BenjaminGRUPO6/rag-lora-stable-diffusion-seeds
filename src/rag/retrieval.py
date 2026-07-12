from __future__ import annotations


def build_retrieval_query(prediction: dict, observations: list[str] | None = None) -> str:
    label = prediction.get("label", "unknown")
    details = "; ".join(observations or [])
    return (
        f"Daño estimado en semilla: {label}. "
        f"Observaciones visuales: {details}. "
        "Recuperar posibles causas, señales relacionadas, prevención, almacenamiento y manejo."
    )
