# Changes Implemented

- `app/app.py` fue reemplazado por `app/streamlit_app.py`.
- `scripts/run_demo.py` ahora usa rutas absolutas y funciona desde otra ubicacion.
- Se agrego fallback RAG local por metadata en `src/rag/retrieval.py` y `src/pipelines/analyze_seed.py`.
- `src/rag/embeddings.py` evita acceso de red por defecto con `local_files_only=True`.
- Se agregaron pruebas para imports, entrypoint, checkpoint disponible/faltante, RAG faltante/disponible, smoke y validaciones funcionales.
- Se generaron resultados reproducibles en `results/audit/`, `results/app_smoke_test/` y `results/end_to_end/`.

No se modificaron datasets originales, imagenes originales, checkpoints ni pesos.
