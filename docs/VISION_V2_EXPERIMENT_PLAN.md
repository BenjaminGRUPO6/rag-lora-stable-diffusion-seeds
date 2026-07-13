# Vision V2 Experiment Plan

## Objetivo

Preparar una infraestructura de experimentacion para mejorar el clasificador visual de semillas de soja sin alterar el baseline funcional ni entrenar modelos en esta etapa.

## Alcance de esta etapa

- Mantener intacto `results/vision/resnet18_baseline`.
- Versionar una copia de artefactos pequenos actuales en `results/vision/resultados_1_baseline`.
- Preparar espacios de trabajo para mejoras en `results/vision/resultados_2_mejoras`.
- Definir un registro reproducible para futuros experimentos.
- No modificar `train`, `validation`, `test`, datasets, checkpoints ni pesos.

## Experimentos previstos

1. `01_metricas_reconciliadas`: reconciliar metricas existentes sin inventar valores.
2. `02_paridad_inferencia`: comprobar paridad entre evaluacion offline e inferencia de la app.
3. `03_recorte_y_calidad`: evaluar recorte, validacion de imagen y calidad de entrada.
4. `04_analisis_errores`: revisar errores por categoria visual y casos ambiguos.
5. `05_resnet18_v2`: preparar un candidato ResNet18 v2, sujeto a entrenamiento posterior.
6. `06_calibracion`: evaluar calibracion de confianza.
7. `07_tta`: evaluar test-time augmentation.
8. `08_comparacion_modelos`: comparar modelos solo con metricas generadas y registradas.
9. `09_gradcam_interfaz`: explorar explicabilidad visual para la interfaz.
10. `10_lora_generativo`: documentar evidencia generativa sin incorporar sinteticos a `train` sin revision humana.

## Reglas de ejecucion

- No entrenar modelos durante esta preparacion.
- No sobrescribir artefactos de `resultados_1_baseline`.
- No copiar checkpoints `.pt`, `.pth`, `.ckpt`, `.safetensors` ni pesos LoRA a resultados versionables.
- Usar rutas relativas y `pathlib` en codigo Python.
- Guardar graficos nuevos con matplotlib, formato PNG, fondo blanco, titulo, nombre del experimento y al menos 180 DPI.
- Tratar `spotted` como categoria visual, no como diagnostico de hongo.

## Salidas esperadas de futuros experimentos

Cada experimento debe registrar:

- `experiment_id`.
- Fecha UTC.
- Commit de Git cuando este disponible.
- Configuracion usada.
- Seed.
- Artefactos generados y hashes cuando corresponda.
- Notas de revision humana cuando aplique.
