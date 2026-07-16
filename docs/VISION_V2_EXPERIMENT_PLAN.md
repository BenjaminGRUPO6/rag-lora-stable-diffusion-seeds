# Vision V2 Experiment Plan

## Objetivo

Preparar una infraestructura de experimentacion para mejorar el clasificador visual de semillas de soja sin alterar el baseline funcional ni entrenar modelos en esta etapa.

## Alcance de esta etapa

- Mantener intacto `results/vision/resnet18_baseline` y `results/vision/resultados_1_baseline`.
- Versionar una copia de artefactos pequenos actuales en `results/vision/resultados_1_baseline`.
- Preparar espacios de trabajo para mejoras en `results/vision/resultados_2_mejoras`.
- Definir un registro reproducible para futuros experimentos.
- No modificar `train`, `validation`, `test` ni imagenes originales. Los checkpoints nuevos quedan fuera de Git.

## Ejecucion registrada - 08 comparacion modelos

La etapa `08_comparacion_modelos` ya fue ejecutada para comparar EfficientNet-B0 con ResNet18 V2.

Problema detectado: el entrenamiento EfficientNet-B0 quedaba limitado por CPU porque `preprocess_image` se ejecutaba dentro de `__getitem__`. Eso repetia por epoca segmentacion, morfologia, componentes conectados y calidad visual.

Correccion:

- cache regenerable en `data/cache/vision_crops/`;
- `compute_quality=false` durante entrenamiento;
- fallback al crop cuadrado de la imagen original si falta un crop;
- `pin_memory=true` en CUDA;
- transferencias `non_blocking=true`;
- `persistent_workers` solo cuando `num_workers > 0`;
- progreso por batch y checkpoints de recuperacion por epoca.

Benchmark y seleccion:

- 128 imagenes, batch 8, NVIDIA GeForce GTX 1050.
- Recorte en tiempo real: 3.85 imagenes/s.
- Cache `num_workers=0`: 105.60 imagenes/s.
- Cache `num_workers=2`: 5.35 imagenes/s en Windows.
- Configuracion seleccionada: batch 8, `num_workers=0`, cache activa.

Resultados:

- EfficientNet-B0: validation macro-F1 0.899532; test macro-F1 0.868306.
- ResNet18 V2: validation macro-F1 0.920701; test macro-F1 0.903329.
- Modelo seleccionado: ResNet18 V2, por validation macro-F1 superior y mejor latencia CUDA.

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
