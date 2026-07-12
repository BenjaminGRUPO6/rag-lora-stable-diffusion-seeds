# Resumen para el equipo

## Decisión final

El sistema no intentará diagnosticar directamente hongos a partir de una imagen. Primero clasificará cinco defectos visibles definidos por un dataset público. Después, el RAG recuperará evidencia técnica relacionada y generará un informe preliminar con fuentes y advertencias.

## Título

**SeedCare-RAG LoRA: Sistema multimodal para clasificar defectos visibles en semillas de soja, recuperar evidencia técnica mediante RAG y ampliar experimentalmente los datos de entrenamiento con Stable Diffusion 1.5 ajustado con LoRA.**

## Qué se entrenará

1. Clasificador visual mediante transferencia de aprendizaje.
2. Adaptador LoRA sobre Stable Diffusion 1.5.
3. Reentrenamiento comparativo del clasificador con imágenes sintéticas aprobadas.

## Qué no se afirmará

- `spotted` no equivale automáticamente a hongo.
- El informe no reemplaza una evaluación fitosanitaria o de laboratorio.
- El sistema no recomendará una acción sin mostrar la evidencia documental recuperada.
