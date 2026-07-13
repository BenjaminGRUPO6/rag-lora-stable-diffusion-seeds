# Release Checklist

- [x] No se desarrollo en `main`.
- [x] No se modifico `data/raw/`.
- [x] No se eliminaron datasets, checkpoints ni pesos.
- [x] `python scripts/check_environment.py` pasa.
- [x] `python -m compileall app src scripts` pasa.
- [x] `python -m pytest -q` pasa.
- [x] `python scripts/run_demo.py` inicia Streamlit sin traceback.
- [x] `python scripts/smoke_test_app.py` devuelve PASS.
- [x] `python scripts/run_functional_test.py` devuelve passed true.
- [x] La app funciona sin API externa obligatoria.
- [x] No se presentan fuentes inventadas.
- [x] No se presenta `spotted` como diagnostico de hongo.
- [x] No se ejecuto entrenamiento.
- [ ] Revisar manualmente `git diff` antes de commit.
