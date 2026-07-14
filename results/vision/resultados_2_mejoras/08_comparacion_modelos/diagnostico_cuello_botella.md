# Diagnostico de cuello de botella - EfficientNet-B0

Resultados 2 - comparacion de modelos.

## Hallazgo principal

El cuello de botella estaba en el pipeline de datos, antes de que los tensores llegaran a CUDA. La ruta V2 usaba `AutoCropTransform` al inicio de `build_v2_transforms`; por tanto cada llamada a `__getitem__` ejecutaba `preprocess_image`. Esa funcion calcula segmentacion por distancia al borde, medias locales, morfologia, componentes conectados, crop y metricas de calidad visual. Como el transform estaba dentro del dataset, el costo se repetia por imagen y por epoca.

El patron observado encaja con GPU casi ociosa, VRAM reservada sin OOM, CPU creciendo y ausencia de checkpoints: el proceso seguia vivo, pero esperaba datos.

## Inventario solicitado

1. Dentro de `__getitem__` se cargaba la imagen con `ImageFolder`, se aplicaba `AutoCropTransform`, luego aumentos realistas, conversion a tensor y normalizacion.
2. Si `auto_crop=true`, el recorte se calculaba cada epoca porque estaba dentro del transform del dataset.
3. El control de calidad se ejecutaba por imagen y por epoca porque `preprocess_image` siempre llamaba `assess_visual_quality`.
4. No se detectaron hashes de contenido durante entrenamiento. El nuevo cache usa hash estable de ruta solo para nombrar crops regenerables.
5. Antes de la correccion no se escribian archivos desde el DataLoader, pero tampoco habia cache. La correccion mantiene esa regla: `__getitem__` no escribe crops.
6. La transformacion costosa era el recorte automatico: mascara adaptativa, morfologia, componentes conectados y calidad. Las transformaciones torchvision restantes son ligeras en comparacion.
7. Con 4,179 imagenes de train y batch size 8, una epoca tiene aproximadamente 523 batches de entrenamiento.
8. `num_workers` estaba en 0 en `configs/vision_v2_efficientnet_b0.yaml` y `configs/vision_v2_resnet18.yaml`.
9. `persistent_workers` no estaba activo. Ahora se activa solo si `num_workers > 0`.
10. `pin_memory` ya dependia de CUDA; se mantiene y queda explicito en los DataLoaders V2.
11. Los tensores se transferian sin `non_blocking`. Ahora se usa `non_blocking=true` cuando CUDA esta activa.
12. AMP usaba `torch.amp.GradScaler("cuda")` y autocast CUDA. Ahora autocast recibe `device_type=device.type` y solo se habilita si el dispositivo es CUDA.
13. El modelo se movia a CUDA con `.to(device)` y los batches tambien, pero la GPU esperaba al DataLoader.
14. `python -u` resuelve buffering externo. El CLI ahora fuerza line buffering en stdout/stderr cuando es posible.
15. No existia progreso por batch porque `run_epoch` solo acumulaba metricas y reportaba al final de la epoca.

## Correccion aplicada

- Se agrego `V2CropImageFolder`, que puede usar `auto_crop`, `cache_preprocessing`, `compute_quality` y `fallback_to_original`.
- Se agrego cache regenerable en `data/cache/vision_crops/`.
- Se agrego `build_crop_cache` para precalcular crops fuera del loop de entrenamiento.
- En entrenamiento, `compute_quality=false` evita calculo de blur y metricas visuales por imagen.
- Con cache activa, `__getitem__` solo lee crops existentes; si falta uno, usa fallback a la imagen original y no escribe archivos.
- Se agrego progreso por batch con `epoch`, `batch`, `total_batches`, perdida, imagenes/segundo, `data_time`, `compute_time`, memoria GPU y tiempo transcurrido.
- Se agregaron flags CLI para workers, auto-crop, cache, max samples, logging, checkpoints por epoca y profiling.

## Evidencia esperada

El benchmark `scripts/benchmark_vision_dataloader.py` compara:

- A: sin recorte automatico.
- B: recorte en tiempo real.
- C: recorte mediante cache.
- D: cache con `num_workers=0`.
- E: cache con `num_workers=2`.

La seleccion runtime debe basarse en ausencia de OOM/bloqueo, mayor `images_per_second`, menor `data_time` y estabilidad en Windows.
