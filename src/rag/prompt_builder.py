from __future__ import annotations

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
