# Modelos

Los pesos, checkpoints y adaptadores entrenados no se versionan en Git.

Ubicaciones locales esperadas:

- `models/vision/`: checkpoints del clasificador visual, por ejemplo ResNet18.
- `models/lora/`: adaptadores LoRA y artefactos de entrenamiento generativo.
- `checkpoints/`: checkpoints temporales o intermedios, si se generan fuera de `models/`.

Para cada artefacto conservado, documenta en los resultados del experimento:

- nombre del archivo;
- arquitectura o modelo base;
- configuracion usada;
- fecha de entrenamiento;
- hash o checksum;
- ubicacion externa segura, si aplica.

Los patrones de exclusion correspondientes se mantienen en `.gitignore`.
