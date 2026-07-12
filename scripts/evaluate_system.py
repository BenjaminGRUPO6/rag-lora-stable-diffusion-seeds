"""Punto de entrada para consolidar métricas.

Se implementará después de obtener predicciones del modelo visual, resultados del RAG
y evaluaciones de imágenes sintéticas.
"""

from pathlib import Path


def main() -> None:
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    print("Complete las matrices de evaluación descritas en docs/06_METRICAS.md")


if __name__ == "__main__":
    main()
