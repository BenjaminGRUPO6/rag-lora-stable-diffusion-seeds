# Vision Results Index

## Directorios

- `resnet18_baseline`: salida funcional existente. No mover, borrar ni reemplazar.
- `resultados_1_baseline`: copia versionable de artefactos pequenos actuales. Estado: `UNRECONCILED`.
- `resultados_2_mejoras`: espacios preparados para experimentos de mejora.

## Resultados 1

La carpeta `resultados_1_baseline` contiene JSON, CSV, PNG y YAML copiados desde `results/vision/resnet18_baseline`, mas `manifest.json` con origen y SHA-256.

No contiene checkpoints `.pt` ni pesos.

## Resultados 2

Subdirectorios preparados:

- `01_metricas_reconciliadas`
- `02_paridad_inferencia`
- `03_recorte_y_calidad`
- `04_analisis_errores`
- `05_resnet18_v2`
- `06_calibracion`
- `07_tta`
- `08_comparacion_modelos`
- `09_gradcam_interfaz`
- `10_lora_generativo`
- `final`

## Estado

No se entrenaron modelos ni se modificaron datasets durante la creacion de esta infraestructura.
