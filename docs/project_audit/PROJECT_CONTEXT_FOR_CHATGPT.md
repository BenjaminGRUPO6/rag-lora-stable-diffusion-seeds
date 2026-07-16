# Contexto autocontenido para ChatGPT

## Nombre y objetivo
`rag-lora-stable-diffusion-seeds`: clasifica defectos visibles de semillas de soja, recupera evidencia tecnica con RAG y documenta LoRA SD 1.5 para ampliacion sintetica futura.

## Estructura
`app/streamlit_app.py` interfaz; `src/pipelines/analyze_seed.py` orquestador; `src/vision/` clasificador; `src/rag/` recuperacion; `src/reports/` informe; `src/synthetic_data/` LoRA; `data/`, `models/`, `vector_db/`, `results/`, `docs/`, `configs/`.

## Cinco clases
`intact`, `spotted`, `immature`, `broken`, `skin_damaged`. `spotted` es categoria visual, no diagnostico de hongo.

## Dataset
Fuente en `data/metadata/dataset_sources.csv`; manifiesto `data/metadata/dataset_split.csv`; total 5513. Splits y clases en `results/project_audit/*counts.csv`.

## Checkpoint y modelo visual
Produccion `resnet18_v2_tta_light`, checkpoint `models/vision/resnet18_v2_best.pt`, config `configs/production_vision_model.yaml` + `configs/vision_v2_resnet18.yaml`, imagen 224 px, normalizacion ImageNet.

## Metricas verificadas
R1 accuracy 0.6704980842911877, macro-F1 0.6259550750897566. R2 final accuracy 0.9176245210727969, macro-F1 0.9168669642726247. Usar `final_metrics.json` como comparativa final.

## Discrepancias
R1 tiene reconciliacion por validacion obsoleta archivada. Detalle en `metrics_consistency.json`.

## Pipeline Streamlit
Upload JPG/PNG -> validacion -> `preprocess_image` -> `run_analysis` -> `analyze_seed` -> ResNet18/TTA -> RAG -> reporte -> tabs/descargas. `app/app.py` no existe.

## Pipeline RAG
`build_retrieval_query` por clase; `FaissRetriever` usa `vector_db`; fallback `MetadataKeywordRetriever`; `generate_preliminary_report` arma informe.

## LoRA
El LoRA es un adaptador generativo para Stable Diffusion. Su función es generar imágenes sintéticas de semillas. No clasifica la imagen cargada, no ejecuta el RAG y no aumenta directamente la confianza del clasificador. Config `configs/lora_sd15.yaml`; adaptador `models/lora/soybean_sd15/pytorch_lora_weights.safetensors`; metadata 1000. Hardware NO VERIFICADA; no usado para reentrenar clasificador.

## Resultados
R1 en `results/vision/resultados_1_baseline/`; R2 en `results/vision/resultados_2_mejoras/`, final en `final/`.

## Pruebas y comandos
Activar `.venv`; ejecutar `python -m pytest -q`, smoke, funcional y `python scripts/run_demo.py --port 8501`.

## Limitaciones y proximos pasos
No entrenar/generar en esta etapa. Limpiar Git. Mantener datos, pesos, indices y `.env` locales. Sinteticos a train solo tras revision humana.
