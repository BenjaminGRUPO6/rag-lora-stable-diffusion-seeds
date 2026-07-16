# Estado del proyecto

Actualizado con evidencia local disponible al 2026-07-14.

## Resultados 2 - comparacion de modelos

- [x] Cuello de botella EfficientNet-B0 diagnosticado: recorte automatico y calidad visual dentro de `__getitem__`.
- [x] Cache regenerable creada en `data/cache/vision_crops/`; no modifica imagenes originales ni splits.
- [x] Benchmark DataLoader completado en 128 imagenes: cache `num_workers=0` fue la opcion estable y mas rapida en Windows.
- [x] Smoke test EfficientNet-B0 en CUDA completado con forward, loss, backward, optimizer step, AMP, validation, checkpoint temporal e historial.
- [x] EfficientNet-B0 entrenada 25 epocas con seed 42, batch 8, `num_workers=0`, AMP y seleccion por validation macro-F1.
- [x] Comparacion controlada con ResNet18 V2 completada; ResNet18 V2 supera a EfficientNet-B0 por validation macro-F1.
- [x] Configuracion final consolidada: `resnet18_v2_tta_light`, seleccionada por validation macro-F1.

Metricas clave:

- EfficientNet-B0 validation macro-F1: 0.899532; test macro-F1: 0.868306; accuracy test: 0.869732.
- ResNet18 V2 validation macro-F1: 0.920701; test macro-F1: 0.903329; accuracy test: 0.904215.
- Modelo de produccion: ResNet18 V2 + TTA light (`resnet18_v2_tta_light`), por mayor validation macro-F1.
- Configuracion final: validation macro-F1 0.926503; test macro-F1 0.916867; test accuracy 0.917625.
- Mejora test macro-F1 vs Resultados 1: 0.290912 absoluta; 46.47% relativa.

## Estado final documentado

- [x] Dataset Soybean Seeds v6 auditado, depurado y dividido.
- [x] Baseline ResNet18 entrenado y evaluado con metrica canonica reconciliada.
- [x] Resultados 1 y Resultados 2 consolidados en `results/vision/resultados_2_mejoras/final/`.
- [x] Corpus documental RAG registrado en `data/metadata/document_sources.csv`.
- [x] Indice FAISS construido en `vector_db/`.
- [x] Evaluacion de recuperacion RAG ejecutada.
- [x] Integracion de demo con ResNet18, RAG e informe preliminar.
- [x] Stable Diffusion 1.5 + LoRA entrenado con evidencia local parcial.
- [x] Evaluacion final de vision consolidada en `results/vision/resultados_2_mejoras/final/final_metrics.json`.
- [x] Experimento B con imagenes sinteticas aplazado como trabajo futuro.

## Metricas canonicas

- Dataset: 5513 imagenes auditadas; 5223 incluidas; 290 duplicados exactos excluidos; split train 4179, validation 522, test 522.
- Vision Resultados 1 test: accuracy 0.670498; macro-F1 0.625955; 522 muestras.
- Vision Resultados 2 final test: accuracy 0.917625; macro-F1 0.916867; 522 muestras.
- RAG recuperacion: 20 consultas; Hit@1 0.45; Hit@3 0.70; Hit@5 0.80; MRR 0.589167.
- Sistema demo: 5 casos ejecutados; 5 exitosos; tiempo total medio 9.671234 s.
- LoRA: adaptador local de 6.12 MB; metadata de 1000 imagenes; estado de evidencia `PARTIAL`.

## Pendientes reales

- Revisar humanamente las 20 consultas RAG para obtener metricas cualitativas.
- Completar evidencia LoRA faltante: logs, hardware, tiempo de entrenamiento, comparacion base vs. LoRA y evaluacion visual humana.
- Ejecutar el Experimento B solo despues de revisar y aprobar imagenes sinteticas; los sinteticos solo pueden entrar en `train`.
- Revisar errores restantes del clasificador final, especialmente casos de alta confianza incorrecta.

## Restricciones de comunicacion

- `spotted` es una categoria visual; no se presenta como diagnostico de hongo.
- La herramienta no es diagnostica ni reemplaza evaluacion especializada.
- No se reportan metricas antiguas marcadas como obsoletas por la reconciliacion.
- No se inventan campos bibliograficos faltantes.
