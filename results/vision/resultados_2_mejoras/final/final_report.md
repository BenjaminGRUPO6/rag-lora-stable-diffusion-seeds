# Consolidacion Resultados 1 vs Resultados 2

Generado UTC: `2026-07-14T06:08:55.358452+00:00`.

## Seleccion final

La configuracion final se selecciona por `validation_macro_f1`; el split `test` se reporta solo como evaluacion final.

- Modelo final: `resnet18_v2_tta_light`.
- Arquitectura: `resnet18`.
- Checkpoint: `models/vision/resnet18_v2_best.pt`.
- TTA: `light` con `2` vistas.
- Temperatura: `1.119302`.
- Validation macro-F1: `0.926503`.
- Test macro-F1 final: `0.916867`.
- Test accuracy final: `0.917625`.

## Comparacion principal

| metrica | Resultados 1 | Resultados 2 final | diferencia absoluta | diferencia porcentual | lectura |
| --- | ---: | ---: | ---: | ---: | --- |
| validation macro-F1 | 0.660513 | 0.926503 | 0.265990 | 40.27% | mejora |
| test macro-F1 | 0.625955 | 0.916867 | 0.290912 | 46.47% | mejora |

## Validaciones

- Mismos splits en candidatos de modelo: `True`.
- Seed registrado o no aplicable: `True`.
- Sin sinteticos en splits del clasificador: `True`.
- Test sin modificar segun soportes y predicciones: `True`.
- Manifest de split: `data/metadata/dataset_split.csv`.

## Candidatos

| candidato | validation macro-F1 | test macro-F1 | decision |
| --- | ---: | ---: | --- |
| `resnet18_baseline` | 0.660513 | 0.625955 | no seleccionado |
| `resnet18_v2` | 0.920701 | 0.903329 | no seleccionado |
| `efficientnet_b0_v2` | 0.899532 | 0.868306 | no seleccionado |
| `resnet18_v2_tta_light` | 0.926503 | 0.916867 | seleccionado |

## Limitaciones

- R1 usa la validacion reconciliada del checkpoint porque los valores altos archivados fueron marcados como obsoletos.
- R1 no registro latencia comparable, por lo que no se afirma mejora de latencia frente a R1.
- TTA mejora la validation macro-F1, pero aumenta la latencia de inferencia de extremo a extremo.
- Las metricas de test son evaluacion final; no se usaron para seleccionar la configuracion de produccion.
- `spotted` es una categoria visual y no un diagnostico de hongo.
- Las imagenes sinteticas no se incorporaron al train del clasificador; cualquier uso futuro requiere revision humana.

## PNG generados

- `results/vision/resultados_2_mejoras/final/r1_vs_r2_dashboard.png`
- `results/vision/resultados_2_mejoras/final/r1_vs_r2_metricas_globales.png`
- `results/vision/resultados_2_mejoras/final/r1_vs_r2_f1_por_clase.png`
- `results/vision/resultados_2_mejoras/final/r1_vs_r2_confianza.png`
- `results/vision/resultados_2_mejoras/final/r1_vs_r2_latencia.png`
- `results/vision/resultados_2_mejoras/final/r2_sistema_final.png`
