# Resumen ejecutivo

## Título

**SeedCare-RAG LoRA: Sistema multimodal para identificar posibles daños físicos, biológicos y morfológicos en semillas, recuperar métodos de prevención y manejo, y ampliar clases minoritarias mediante Stable Diffusion 1.5 ajustado con LoRA.**

## Resumen

El proyecto propone un sistema de apoyo al control de calidad de semillas. Un modelo visual ajustado analiza una fotografía y estima una categoría de daño visible. El resultado activa una consulta en una base RAG compuesta por documentos técnicos verificados. El sistema recupera fragmentos sobre causas, señales relacionadas, prevención, almacenamiento y manejo, y genera un informe con fuentes y limitaciones. Para demostrar entrenamiento generativo y mejorar el balance de datos, se entrena un adaptador LoRA sobre Stable Diffusion 1.5. Las imágenes sintéticas se someten a revisión humana y se incorporan solo a entrenamiento. El desempeño se evalúa con métricas del clasificador, recuperación documental y calidad/fidelidad del aumento sintético.
