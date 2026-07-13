# Changes Implemented

- `app/app.py` fue reemplazado por `app/streamlit_app.py`.
- `scripts/run_demo.py` ahora usa rutas absolutas, puerto 8501 por defecto y mantiene Streamlit activo hasta `Ctrl+C`.
- `scripts/run_demo.py` funciona desde otra ubicacion porque calcula `PROJECT_ROOT` con `pathlib` y ejecuta con `cwd` en la raiz del repositorio.
- `scripts/smoke_test_app.py` falla si stdout/stderr contienen `ModuleNotFoundError`, `No module named 'app'`, `Traceback` o `ImportError`, y valida excepciones de `AppTest`.
- Se agrego fallback RAG local por metadata en `src/rag/retrieval.py` y `src/pipelines/analyze_seed.py`.
- `src/rag/embeddings.py` evita acceso de red por defecto con `local_files_only=True`.
- Se agregaron pruebas para imports, entrypoint, ausencia de `app/app.py`, ejecucion desde otro directorio, checkpoint disponible/faltante, RAG faltante/disponible, smoke y validaciones funcionales.
- Se generaron resultados reproducibles en `results/audit/`, `results/app_smoke_test/` y `results/end_to_end/`.

No se modificaron datasets originales, imagenes originales, checkpoints ni pesos.
