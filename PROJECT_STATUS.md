# Estado del proyecto

Actualizado con evidencia local disponible al 2026-07-13.

## Estado final documentado

- [x] Dataset Soybean Seeds v6 auditado, depurado y dividido.
- [x] Baseline ResNet18 entrenado y evaluado con metrica canonica reconciliada.
- [x] Corpus documental RAG registrado en `data/metadata/document_sources.csv`.
- [x] Indice FAISS construido en `vector_db/`.
- [x] Evaluacion de recuperacion RAG ejecutada.
- [x] Integracion de demo con ResNet18, RAG e informe preliminar.
- [x] Stable Diffusion 1.5 + LoRA entrenado con evidencia local parcial.
- [x] Evaluacion final consolidada en `results/system/final_metrics.json`.
- [x] Experimento B con imagenes sinteticas aplazado como trabajo futuro.

## Metricas canonicas

- Dataset: 5513 imagenes auditadas; 5223 incluidas; 290 duplicados exactos excluidos; split train 4179, validation 522, test 522.
- Vision ResNet18 test: accuracy 0.670498; macro-F1 0.625955; 522 muestras.
- RAG recuperacion: 20 consultas; Hit@1 0.45; Hit@3 0.70; Hit@5 0.80; MRR 0.589167.
- Sistema demo: 5 casos ejecutados; 5 exitosos; tiempo total medio 9.671234 s.
- LoRA: adaptador local de 6.12 MB; metadata de 1000 imagenes; estado de evidencia `PARTIAL`.

## Pendientes reales

- Revisar humanamente las 20 consultas RAG para obtener metricas cualitativas.
- Completar evidencia LoRA faltante: logs, hardware, tiempo de entrenamiento, comparacion base vs. LoRA y evaluacion visual humana.
- Ejecutar el Experimento B solo despues de revisar y aprobar imagenes sinteticas; los sinteticos solo pueden entrar en `train`.
- Mejorar el clasificador, especialmente la clase `intact`, cuyo F1 en test es 0.296296.

## Restricciones de comunicacion

- `spotted` es una categoria visual; no se presenta como diagnostico de hongo.
- La herramienta no es diagnostica ni reemplaza evaluacion especializada.
- No se reportan metricas antiguas marcadas como obsoletas por la reconciliacion.
- No se inventan campos bibliograficos faltantes.
