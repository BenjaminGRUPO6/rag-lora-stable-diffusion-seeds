# Results Versioning

## Nomenclatura obligatoria

- Resultados 1: `results/vision/resultados_1_baseline`
- Resultados 2: `results/vision/resultados_2_mejoras`

## Resultados 1

`resultados_1_baseline` es una copia versionable de artefactos pequenos del baseline actual. Su estado inicial es `UNRECONCILED` porque existe una discrepancia de metricas pendiente de reconciliacion.

Reglas:

- No sobrescribir archivos existentes.
- No agregar checkpoints ni pesos.
- No reinterpretar metricas historicas como metricas nuevas.
- Usar `manifest.json` como fuente de origen, fecha y SHA-256 de cada archivo copiado.

## Resultados 2

`resultados_2_mejoras` contiene subdirectorios para experimentos futuros. Cada experimento debe escribir en su propia carpeta y registrar metadatos con `src.experiments.result_registry`.

## Politica de artefactos

Se pueden versionar:

- JSON pequeno.
- CSV pequeno.
- PNG generado como evidencia.
- YAML de configuracion.
- Markdown de documentacion.

No se deben versionar:

- Checkpoints.
- Datasets.
- Indices vectoriales.
- Caches.
- Tokens o archivos `.env`.
- Pesos LoRA.

## Graficos

Todo grafico nuevo debe:

- Generarse con matplotlib.
- Guardarse como PNG.
- Usar fondo blanco.
- Incluir titulo.
- Incluir el nombre del experimento.
- Guardarse con al menos 180 DPI.
