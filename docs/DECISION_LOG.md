# Decision Log

## Experimento B aplazado

Decision: El segundo entrenamiento de ResNet18 con imagenes sinteticas se aplaza como trabajo futuro. El proyecto conserva el baseline ResNet18 y evalua el LoRA directamente.

Motivo: La integracion final actual debe reconciliar el baseline ResNet18 entrenado y la evidencia del LoRA sin incorporar datos sinteticos al entrenamiento del clasificador.

Alcance: No se entrenan modelos adicionales, no se generan imagenes y no se incorporan imagenes sinteticas a `train` sin revision humana.
