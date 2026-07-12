# Cumplimiento del requisito de entrenamiento

## Entrenamiento 1: modelo visual

Se ajusta un modelo preentrenado de visión sobre las categorías del proyecto. Se actualizan pesos, se guarda un checkpoint propio y se reportan pérdidas, macro-F1, precisión, recall y matriz de confusión.

## Entrenamiento 2: Stable Diffusion 1.5 con LoRA

Se utiliza el modelo base Stable Diffusion 1.5 y se agregan matrices LoRA entrenables en el UNet. El resto del modelo permanece mayormente congelado. El resultado es un adaptador `.safetensors` propio del equipo. Se documentan datos, captions, pasos, learning rate, rank, pérdidas y muestras de validación.

## Experimento principal

1. Clasificador entrenado solo con datos reales.
2. SD 1.5 base frente a SD 1.5 + LoRA.
3. Clasificador entrenado con datos reales más sintéticos aceptados.
4. Comparación objetiva de macro-F1 y F1 de clases minoritarias.

Esta comparación evidencia si el entrenamiento generativo aporta valor o introduce ruido.
