"""
2_test_inferencia_sinteticas.py
================================
Clasifica todas las imágenes sintéticas en ``experimentacion/sinteticas_crudas/``
usando el modelo ResNet18 V2 entrenado (con calibración de temperatura) y
guarda un reporte detallado en ``experimentacion/resultados_inferencia/``.

Uso:
    .venv\\Scripts\\python.exe experimentacion/2_test_inferencia_sinteticas.py

Salidas:
    experimentacion/resultados_inferencia/predicciones.csv
    experimentacion/resultados_inferencia/metricas_por_clase.json
    experimentacion/resultados_inferencia/resumen.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Añadir la raíz del proyecto al sys.path para importar src
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.vision.inference_engine import VisionInferenceEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
SINTETICAS_DIR = PROJECT_ROOT / "experimentacion" / "sinteticas_crudas"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "vision" / "resnet18_v2_best.pt"
TEMPERATURE_PATH = PROJECT_ROOT / "models" / "vision" / "resnet18_v2_temperature.json"
OUTPUT_DIR = PROJECT_ROOT / "experimentacion" / "resultados_inferencia"

CLASES_ESPERADAS: list[str] = ["intact", "spotted", "immature", "broken", "skin_damaged"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cargar_motor() -> VisionInferenceEngine:
    """Carga el motor de inferencia ResNet18 V2 con calibración."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"No se encontró el checkpoint: {CHECKPOINT_PATH}")

    engine = VisionInferenceEngine.from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        device=device,
        temperature_path=TEMPERATURE_PATH if TEMPERATURE_PATH.exists() else None,
    )
    temp = engine.temperature
    print(f"Modelo cargado. Clases: {engine.labels}")
    print(f"Calibración de temperatura: {temp:.4f}" if temp else "Sin calibración.")
    return engine


def recolectar_imagenes() -> list[dict]:
    """Recorre sinteticas_crudas y devuelve lista de {clase, ruta}."""
    registros: list[dict] = []
    for clase_dir in sorted(SINTETICAS_DIR.iterdir()):
        if not clase_dir.is_dir():
            continue
        clase = clase_dir.name
        imagenes = sorted(clase_dir.glob("*.jpg")) + sorted(clase_dir.glob("*.png"))
        if not imagenes:
            print(f"  [AVISO] Sin imágenes en clase '{clase}'")
            continue
        for img_path in imagenes:
            registros.append({"clase_real": clase, "ruta": img_path})
    return registros


def predecir(engine: VisionInferenceEngine, registros: list[dict]) -> list[dict]:
    """Ejecuta inferencia sobre cada imagen y devuelve los resultados."""
    resultados: list[dict] = []
    total = len(registros)
    for i, reg in enumerate(registros, 1):
        ruta: Path = reg["ruta"]
        clase_real: str = reg["clase_real"]
        print(f"[{i:>3}/{total}] {clase_real}/{ruta.name}", end=" ... ")
        try:
            pred = engine.predict(ruta)
            correcto = pred.label == clase_real
            print(f"→ {pred.label} ({pred.confidence:.1%}) {'✓' if correcto else '✗'}")
            resultados.append(
                {
                    "archivo": ruta.name,
                    "clase_real": clase_real,
                    "prediccion": pred.label,
                    "correcto": correcto,
                    "confianza": round(pred.confidence, 4),
                    "segunda_clase": pred.second_class,
                    "segunda_confianza": round(pred.second_confidence or 0.0, 4),
                    "margen_top1_top2": round(pred.top1_top2_margin, 4),
                    "calibracion_aplicada": pred.calibration_applied,
                    **{f"prob_{c}": round(pred.probabilities.get(c, 0.0), 4) for c in CLASES_ESPERADAS},
                }
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            resultados.append(
                {
                    "archivo": ruta.name,
                    "clase_real": clase_real,
                    "prediccion": "ERROR",
                    "correcto": False,
                    "confianza": 0.0,
                    "segunda_clase": None,
                    "segunda_confianza": 0.0,
                    "margen_top1_top2": 0.0,
                    "calibracion_aplicada": False,
                    **{f"prob_{c}": 0.0 for c in CLASES_ESPERADAS},
                }
            )
    return resultados


def calcular_metricas(df: pd.DataFrame) -> dict:
    """Calcula accuracy y métricas por clase."""
    total = len(df)
    correctos = int(df["correcto"].sum())
    accuracy_global = correctos / total if total > 0 else 0.0

    metricas_por_clase: dict = {}
    for clase in CLASES_ESPERADAS:
        subset = df[df["clase_real"] == clase]
        n = len(subset)
        if n == 0:
            continue
        aciertos = int(subset["correcto"].sum())
        confianza_media = float(subset["confianza"].mean())
        # Ejemplos mal clasificados con su predicción
        errores = subset[~subset["correcto"]][["archivo", "prediccion", "confianza"]].to_dict("records")
        metricas_por_clase[clase] = {
            "total_imagenes": n,
            "correctas": aciertos,
            "accuracy": round(aciertos / n, 4),
            "confianza_media": round(confianza_media, 4),
            "errores": errores,
        }

    return {
        "total_imagenes": total,
        "correctas": correctos,
        "accuracy_global": round(accuracy_global, 4),
        "por_clase": metricas_por_clase,
    }


def guardar_resultados(df: pd.DataFrame, metricas: dict) -> None:
    """Guarda CSV de predicciones, JSON de métricas y JSON de resumen."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "predicciones.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nPredicciones guardadas en: {csv_path}")

    metricas_path = OUTPUT_DIR / "metricas_por_clase.json"
    with metricas_path.open("w", encoding="utf-8") as fh:
        json.dump(metricas, fh, ensure_ascii=False, indent=2)
    print(f"Métricas guardadas en: {metricas_path}")

    resumen_path = OUTPUT_DIR / "resumen.json"
    resumen = {
        "accuracy_global": metricas["accuracy_global"],
        "total_imagenes": metricas["total_imagenes"],
        "correctas": metricas["correctas"],
        "checkpoint": str(CHECKPOINT_PATH),
        "por_clase": {
            clase: {
                "accuracy": datos["accuracy"],
                "confianza_media": datos["confianza_media"],
                "total": datos["total_imagenes"],
            }
            for clase, datos in metricas["por_clase"].items()
        },
    }
    with resumen_path.open("w", encoding="utf-8") as fh:
        json.dump(resumen, fh, ensure_ascii=False, indent=2)
    print(f"Resumen guardado en: {resumen_path}")


def imprimir_tabla(metricas: dict) -> None:
    """Imprime un resumen visual en consola."""
    print("\n" + "=" * 60)
    print("  RESULTADOS DE INFERENCIA — IMÁGENES SINTÉTICAS")
    print("=" * 60)
    print(f"  Accuracy global : {metricas['accuracy_global']:.1%}  "
          f"({metricas['correctas']}/{metricas['total_imagenes']})")
    print("-" * 60)
    print(f"  {'Clase':<15} {'Total':>6} {'Correctas':>10} {'Accuracy':>10} {'Conf. media':>12}")
    print("-" * 60)
    for clase, datos in metricas["por_clase"].items():
        print(
            f"  {clase:<15} {datos['total_imagenes']:>6} {datos['correctas']:>10} "
            f"{datos['accuracy']:>10.1%} {datos['confianza_media']:>12.1%}"
        )
    print("=" * 60)

    # Errores notables
    for clase, datos in metricas["por_clase"].items():
        if datos["errores"]:
            print(f"\n  Errores en '{clase}':")
            for err in datos["errores"]:
                print(f"    {err['archivo']} → predicho como '{err['prediccion']}' "
                      f"(confianza: {err['confianza']:.1%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  TEST DE INFERENCIA — IMÁGENES SINTÉTICAS CRUDAS")
    print("=" * 60)

    if not SINTETICAS_DIR.exists():
        print(f"Error: No existe el directorio {SINTETICAS_DIR}")
        print("Primero ejecuta: python experimentacion/1_generar_sinteticas.py")
        return

    print("\n[1/4] Cargando motor de inferencia ResNet18 V2...")
    engine = cargar_motor()

    print("\n[2/4] Recolectando imágenes sintéticas...")
    registros = recolectar_imagenes()
    if not registros:
        print("No se encontraron imágenes en sinteticas_crudas/. Verifica la generación.")
        return
    print(f"  → {len(registros)} imágenes encontradas en {len(set(r['clase_real'] for r in registros))} clases.")

    print("\n[3/4] Ejecutando inferencia...")
    resultados = predecir(engine, registros)

    print("\n[4/4] Calculando métricas y guardando resultados...")
    df = pd.DataFrame(resultados)
    metricas = calcular_metricas(df)
    guardar_resultados(df, metricas)
    imprimir_tabla(metricas)


if __name__ == "__main__":
    main()
