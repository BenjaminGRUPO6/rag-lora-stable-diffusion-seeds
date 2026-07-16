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

## Resultados 2 - 08 comparacion modelos

Estado: completado.

Artefactos clave:

- Diagnostico: `resultados_2_mejoras/08_comparacion_modelos/diagnostico_cuello_botella.md`
- Benchmark DataLoader: `dataloader_benchmark.csv`
- Configuracion runtime: `efficientnet_runtime_config.json`
- EfficientNet-B0: metricas validation/test, historial, predicciones, reporte de clasificacion y PNG `r2_efficientnet_*`
- Comparacion: `model_comparison.csv`, `model_selection.json`, `latency_comparison.csv`, `parameter_comparison.csv`
- Produccion: `configs/production_vision_model.yaml` selecciona ResNet18 V2 para Resultados 2

Resumen:

- Cuello de botella: recorte automatico y calidad visual dentro del DataLoader.
- Configuracion elegida: batch 8, `num_workers=0`, cache activa.
- EfficientNet-B0: validation macro-F1 0.899532; test macro-F1 0.868306.
- ResNet18 V2: validation macro-F1 0.920701; test macro-F1 0.903329.
- Modelo seleccionado: ResNet18 V2. No se reemplaza produccion con EfficientNet-B0 porque la evidencia no lo justifica.

## Estado

La infraestructura inicial no entreno modelos. La etapa 08 posterior si entreno EfficientNet-B0 y genero resultados de comparacion sin modificar Resultados 1, splits ni imagenes originales.
