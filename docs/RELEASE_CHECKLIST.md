# Release Checklist

- [x] No se desarrollo en `main`.
- [x] No se modifico `data/raw/`.
- [x] No se eliminaron datasets, checkpoints ni pesos.
- [x] `python scripts/check_environment.py` pasa.
- [x] `python -m compileall app src scripts` pasa.
- [x] `python -m pytest -q` pasa.
- [x] `python scripts/run_demo.py --port 8501` inicia Streamlit sin traceback y permanece activo hasta interrupcion.
- [x] `python -m streamlit run app/streamlit_app.py --server.port 8501` inicia sin configurar `PYTHONPATH`.
- [x] `python scripts/smoke_test_app.py` devuelve PASS.
- [x] `python scripts/run_functional_test.py` devuelve passed true.
- [x] `streamlit.testing.v1.AppTest` renderiza `SeedCare-RAG` sin excepciones.
- [x] stdout/stderr de Streamlit no contienen `ModuleNotFoundError`, `No module named 'app'`, `Traceback` ni `ImportError`.
- [x] La app funciona sin API externa obligatoria.
- [x] No se presentan fuentes inventadas.
- [x] No se presenta `spotted` como diagnostico de hongo.
- [x] No se ejecuto entrenamiento.
- [ ] Revisar manualmente `git diff` antes de commit.
