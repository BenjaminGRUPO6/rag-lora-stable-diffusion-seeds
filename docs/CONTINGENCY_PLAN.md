# Plan de contingencia

## Riesgo: Streamlit no inicia

Acciones:

1. Ejecutar prueba de arranque:

```powershell
python scripts/run_demo.py --timeout 40
```

2. Usar CLI:

```powershell
python scripts/analyze_seed.py `
  --image data/processed/validation/immature/1.jpg `
  --output results/reports/demo_cli_report.json `
  --device cpu
```

3. Mostrar `results/system/demo_cases.csv` y `docs/FINAL_RESULTS.md`.

## Riesgo: falta checkpoint visual

Sintoma: error de recurso faltante para `models/vision/resnet18_baseline_best.pt`.

Acciones:

- Explicar que los pesos no se versionan.
- Mostrar resultados ya consolidados en `results/vision/resnet18_baseline/`.
- No descargar pesos durante la sustentacion.

## Riesgo: falta indice FAISS

Sintoma: falta `vector_db/index.faiss` o `vector_db/metadata.json`.

Acciones:

```powershell
python scripts/build_vector_db.py `
  --config configs/rag.yaml `
  --documents data/documents/accepted `
  --sources data/metadata/document_sources.csv `
  --output vector_db
```

Si no hay tiempo, mostrar `results/rag/evaluation/evaluation_report.md`.

## Riesgo: inferencia muy lenta

Acciones:

- Usar una imagen pequena validada.
- Ejecutar con `--device cpu` si CUDA falla.
- Explicar que el primer caso incluye carga en frio.
- Mostrar `results/system/latency_report.csv`.

## Riesgo: prediccion incorrecta durante demo

Acciones:

- No ocultar el resultado.
- Mostrar confianza, incertidumbre y top probabilidades.
- Enfatizar que el sistema es preliminar y requiere revision humana.
- Usar el caso para explicar limitaciones del clasificador.

## Riesgo: RAG recupera fuente no esperada

Acciones:

- Mostrar que la evaluacion ya registro cuatro fallos en Hit@5.
- Explicar que RAG recupera evidencia documental, no garantiza verdad completa.
- Usar `results/rag/evaluation/query_results.csv` para trazabilidad.

## Riesgo: preguntas sobre LoRA

Respuesta base:

"LoRA fue entrenado y existe un adaptador local, pero la evidencia es parcial. No hay comparacion base vs. LoRA ni evaluacion humana; por eso no afirmamos mejora del clasificador con sinteticos."

## Riesgo: preguntas sobre diagnostico de hongos

Respuesta base:

"No diagnosticamos hongos. `spotted` es una categoria visual del dataset. Cualquier posible causa se trata como informacion documental que requiere evaluacion especializada."
