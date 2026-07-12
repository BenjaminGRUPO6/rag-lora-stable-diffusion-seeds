from __future__ import annotations


def build_report_prompt(prediction: dict, retrieved: list[dict]) -> str:
    sources = "\n\n".join(
        f"Fuente {i+1}: {item.get('title', 'Sin título')}\n{item.get('text', '')}"
        for i, item in enumerate(retrieved)
    )
    return f"""
Redacta un informe preliminar en español.

Resultado visual estimado:
- Clase: {prediction.get('label')}
- Confianza: {prediction.get('confidence', 0):.2%}

Contexto recuperado:
{sources}

Reglas:
1. No afirmar un diagnóstico definitivo.
2. Diferenciar observación, posible causa y recomendación.
3. Citar la fuente usada para cada afirmación relevante.
4. Incluir limitaciones y recomendación de revisión especializada.
""".strip()
