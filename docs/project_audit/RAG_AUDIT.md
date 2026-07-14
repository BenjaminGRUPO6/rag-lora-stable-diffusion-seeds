# Auditoria RAG

Auditoria: 2026-07-14T02:12:47-05:00

- Config: `configs/rag.yaml`.
- Fuentes: `6`.
- Chunks: `1316`.
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Indice: `vector_db/index.faiss`; metadata: `vector_db/metadata.json`.
- top-k: `5`.
- Fallback: `MetadataKeywordRetriever`.

El RAG recupera evidencia; no clasifica imagenes.
