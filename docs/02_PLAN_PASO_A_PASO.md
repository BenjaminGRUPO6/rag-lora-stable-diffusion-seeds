# Plan de ejecución paso a paso

## Fase 0 - GitHub y entorno

- Crear repositorio privado `rag-lora-stable-diffusion-seeds`.
- Invitar colaboradores.
- Clonar, crear `.venv` y ejecutar pruebas.
- Trabajar con ramas y Pull Requests.

## Fase 1 - Dataset

- Auditar imágenes sin modificarlas.
- Definir etiquetas y subetiquetas.
- Revisar manualmente casos dudosos.
- Separar 80/10/10 antes del aumento.

## Fase 2 - Modelo visual

- Entrenar ResNet18 con pesos preentrenados.
- Guardar línea base, historial y matriz de confusión.
- Analizar errores por clase.

## Fase 3 - Stable Diffusion 1.5 + LoRA

- Elegir una o dos clases minoritarias.
- Preparar imágenes y captions con trigger word.
- Realizar prueba de 50 pasos.
- Entrenar aproximadamente 800 pasos en Colab.
- Generar muestras y revisarlas.
- Incorporar solo muestras aceptadas a train.
- Reentrenar el clasificador y comparar.

## Fase 4 - RAG

- Reunir fuentes verificables.
- Registrar metadatos.
- Crear chunks, embeddings e índice FAISS.
- Construir consultas desde la predicción visual.
- Evaluar recuperación con preguntas conocidas.

## Fase 5 - Integración

- Conectar imagen, predicción, recuperación e informe.
- Mostrar confianza, fuentes y limitaciones.
- Preparar entradas de demostración y video de respaldo.

## Fase 6 - Evaluación e informe

- Consolidar métricas.
- Comparar líneas base.
- Documentar limitaciones y ética.
- Preparar informe IEEE y presentación.
