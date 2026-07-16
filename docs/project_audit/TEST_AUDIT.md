# Auditoria de pruebas

Auditoria: 2026-07-14T02:12:47-05:00

- `python -m compileall app src scripts`: PASS en `.venv`.
- `python -m pytest -q`: PASS en `.venv`, 130 pruebas, 0 fallos, 1 warning.
- `python scripts/check_environment.py`: PASS en `.venv`, FAIL global.
- Smoke test: PASS, HTTP 200, titulo renderizado.
- Funcional: PASS, checkpoint/modelo/RAG/reporte validados.
