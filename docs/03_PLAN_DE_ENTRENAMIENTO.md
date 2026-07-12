# Plan de entrenamiento

## Entrenamiento A: clasificador visual

- Arquitectura inicial: ResNet18 preentrenada.
- Sustituir la capa final por cinco salidas.
- Experimento base: imágenes reales con aumento convencional.
- Métricas: accuracy, macro-F1, F1 por clase, matriz de confusión y pérdida.
- Selección del checkpoint por macro-F1 de validación.

## Entrenamiento B: Stable Diffusion 1.5 + LoRA

- Preparar un subconjunto de imágenes y captions.
- Ejecutar una prueba de 100 pasos.
- Revisar memoria, carga y calidad.
- Ejecutar una corrida completa inicial de 800 pasos.
- Guardar adaptador, configuración y prompts de validación.
- Generar imágenes por clase y someterlas a revisión humana.

## Entrenamiento C: comparación con datos sintéticos

- Modelo 1: entrenamiento con imágenes reales.
- Modelo 2: mismas imágenes reales más sintéticas aprobadas.
- Mantener idénticos `validation` y `test`.
- Comparar macro-F1 y F1 por clase.

## Evidencias para la sustentación

- Código y configuración.
- Curvas de pérdida.
- Checkpoints y adaptador LoRA.
- Comparación visual antes/después.
- Tabla de métricas.
- Registro de imágenes sintéticas aceptadas y rechazadas.
