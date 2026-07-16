# Resultados finales

Todas las metricas provienen de artefactos locales. Fuente consolidada de vision: `results/vision/resultados_2_mejoras/final/final_metrics.json`.

## Dataset

| metrica | valor | fuente |
| --- | ---: | --- |
| imagenes auditadas | 5513 | `results/dataset_preparation/summary.json` |
| imagenes incluidas | 5223 | `results/dataset_preparation/summary.json` |
| imagenes excluidas | 290 | `results/dataset_preparation/summary.json` |
| razon de exclusion | duplicado exacto | `results/dataset_preparation/summary.json` |
| train | 4179 | `results/dataset_preparation/summary.json` |
| validation | 522 | `results/dataset_preparation/summary.json` |
| test | 522 | `results/dataset_preparation/summary.json` |
| sinteticos en train | 0 | `results/dataset_preparation/summary.json` |

## ResNet18

El baseline ResNet18 fue entrenado. La evaluacion canonica reconciliada usa `models/vision/resnet18_baseline_best.pt`, `data/metadata/dataset_split.csv` y `results/vision/resnet18_baseline/metrics_test.json`.

| metrica | valor |
| --- | ---: |
| muestras test | 522 |
| accuracy | 0.670498 |
| macro precision | 0.741193 |
| macro recall | 0.650534 |
| macro-F1 | 0.625955 |

| clase | support | F1 |
| --- | ---: | ---: |
| intact | 91 | 0.296296 |
| spotted | 106 | 0.724409 |
| immature | 112 | 0.688406 |
| broken | 100 | 0.641975 |
| skin_damaged | 113 | 0.778689 |

Nota de reconciliacion: `results/vision/resnet18_baseline/reconciliation_report.md` marca como obsoleto un `test_macro_f1` alto de un resumen anterior. El resultado final usa `macro-F1=0.625955`.

## Resultados 2 - seleccion final de produccion

La comparacion controlada usa los mismos splits reales, seed 42, cinco clases visuales y seleccion por validation macro-F1. El test se reporta despues de elegir el mejor checkpoint por validation.

| modelo/configuracion | validation macro-F1 | test macro-F1 | test accuracy | recall intact | F1 intact | recall broken | F1 broken | latencia |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet18 V2 | 0.920701 | 0.903329 | 0.904215 | 0.923077 | 0.879581 | 0.890000 | 0.898990 | 7.052 ms CUDA |
| EfficientNet-B0 | 0.899532 | 0.868306 | 0.869732 | 0.912088 | 0.842640 | 0.780000 | 0.834225 | 19.764 ms CUDA |
| ResNet18 V2 + TTA light | 0.926503 | 0.916867 | 0.917625 | 0.945055 | 0.895833 | 0.900000 | 0.909091 | 0.287293 s/img test |

Seleccion final: `resnet18_v2_tta_light`. Se selecciona por validation macro-F1. El test macro-F1 de 0.916867 se reporta unicamente como evaluacion final. Frente a Resultados 1, la mejora test macro-F1 es 0.290912 absoluta y 46.47% relativa. No se afirma mejora de latencia frente a R1 porque R1 no registro latencia comparable.

Rendimiento del pipeline EfficientNet-B0:

- Hardware: NVIDIA GeForce GTX 1050.
- Cuello de botella inicial: recorte automatico y calidad visual dentro de `__getitem__`.
- Correccion: cache regenerable en `data/cache/vision_crops/`, `compute_quality=false` durante entrenamiento, `pin_memory=true`, transferencias `non_blocking` y progreso por batch.
- Benchmark 128 imagenes: recorte en tiempo real 3.85 imagenes/s; cache con `num_workers=0` 105.60 imagenes/s; cache con `num_workers=2` 5.35 imagenes/s.
- Configuracion seleccionada: batch 8, `num_workers=0`, cache activa.
- Entrenamiento EfficientNet-B0: 25 epocas, 2122.40 s, mejor epoca 25.

Artefactos principales: `results/vision/resultados_2_mejoras/final/`.

## RAG

La evaluacion RAG mide recuperacion documental, no generacion por LLM.

| metrica | valor |
| --- | ---: |
| consultas | 20 |
| Hit@1 | 0.450000 |
| Hit@3 | 0.700000 |
| Hit@5 | 0.800000 |
| MRR | 0.589167 |
| Precision@1 | 0.500000 |
| Precision@3 | 0.472222 |
| Precision@5 | 0.433333 |
| consultas evaluables para precision | 12 |
| latencia media de recuperacion | 12.807 ms |

Consultas fallidas en Hit@5: `RAG009`, `RAG012`, `RAG017`, `RAG020`.

Revision humana: pendiente para 20 consultas.

## LoRA

Stable Diffusion 1.5 + LoRA fue entrenado y existe evidencia local parcial.

| evidencia | valor |
| --- | --- |
| estado | `PARTIAL` |
| adaptador local | `models/lora/soybean_sd15/pytorch_lora_weights.safetensors` |
| tamano | 6.12 MB |
| metadata | 1000 registros |
| distribucion | 200 imagenes por clase visual |
| modelo base | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| rank | 8 |
| learning rate | 0.0001 |
| resolucion | 512 |
| mixed precision | `fp16` |
| seed | 42 |

Faltantes: logs del notebook, hardware, tiempo de entrenamiento, comparacion base vs. LoRA, evaluacion visual humana y metricas de aceptacion de sinteticos.

## Sistema integrado

| metrica | valor |
| --- | ---: |
| casos demo | 5 |
| casos exitosos | 5 |
| inferencia visual media | 8.850886 s |
| recuperacion media | 0.809960 s |
| tiempo total medio | 9.671234 s |

Los tiempos incluyen un primer caso con carga en frio.

## Tabla final de cumplimiento

| requisito | evidencia | archivo | estado |
| --- | --- | --- | --- |
| Clasificar defectos visibles en cinco categorias | ResNet18 V2 + TTA light seleccionado por validation | `results/vision/resultados_2_mejoras/final/final_metrics.json` | Cumplido con limitaciones |
| Usar datos reales auditados | 5513 auditadas, 5223 incluidas | `results/dataset_preparation/summary.json` | Cumplido |
| Mantener `data/raw/` inmutable | Preparacion copia a `data/processed/` | `src/data/cleaning.py`, `src/data/split_dataset.py` | Cumplido |
| Recuperar evidencia documental | FAISS con 1316 chunks y 6 documentos | `vector_db/metadata.json` | Cumplido |
| Evaluar RAG | 20 consultas, Hit@5 0.80 | `results/rag/evaluation/metrics.json` | Cumplido parcialmente |
| Generar informe con fuentes y limitaciones | Reporte deterministico integrado | `src/reports/report_generator.py` | Cumplido |
| Entrenar LoRA | Adaptador local y metadata de 1000 registros | `results/lora/training_summary.json` | Cumplido con evidencia parcial |
| No usar sinteticos en evaluacion | `synthetic_train_images=0`; test real | `results/dataset_preparation/summary.json` | Cumplido |
| Aplazar Experimento B | Decision documentada | `docs/DECISION_LOG.md` | Cumplido |
| No diagnosticar hongos | Advertencias en app y reportes | `app/components/demo_helpers.py`, `src/reports/report_generator.py` | Cumplido |
| Preparar sustentacion | Guion, contingencia y Q&A | `docs/DEMO_SCRIPT_15_MINUTES.md`, `docs/PANEL_QUESTIONS_AND_ANSWERS.md` | Cumplido |

## Interpretacion

El sistema demuestra integracion funcional, pero no debe presentarse como producto diagnostico. El clasificador aun puede fallar, el RAG requiere revision humana y LoRA necesita evidencia adicional para sostener conclusiones sobre calidad generativa o mejora del clasificador.
