# Plan del RAG y los informes

## Corpus

Artículos, manuales, fichas técnicas y guías sobre defectos de semillas, daños físicos, inmadurez, manchas, almacenamiento, humedad y control de calidad.

## Pipeline

1. Extraer texto y metadatos.
2. Dividir en fragmentos.
3. Crear embeddings.
4. Construir índice FAISS.
5. Recuperar top-k fragmentos según categoría y observaciones.
6. Generar un informe únicamente con información respaldada.
7. Mostrar fuente, título y fragmento.

## Evaluación

- Recall@k y Precision@k.
- MRR o nDCG.
- Porcentaje de afirmaciones respaldadas.
- Tasa de respuestas abstentivas cuando no hay evidencia.
- Revisión humana de claridad y fidelidad.
