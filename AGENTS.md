# AGENTS.md

## Contexto del repositorio

Este proyecto clasifica defectos visibles en semillas de soja, recupera evidencia técnica mediante RAG y entrena Stable Diffusion 1.5 con LoRA para un experimento de ampliación sintética.

## Reglas obligatorias para Codex

- Inspecciona el repositorio antes de editar y presenta un plan breve.
- Trabaja únicamente en la etapa solicitada; no adelantes módulos.
- Nunca elimines, renombres ni modifiques archivos dentro de `data/raw/`.
- No subas dataset, pesos, índices vectoriales, cachés, tokens ni archivos `.env`.
- Usa `pathlib`, type hints, docstrings y rutas relativas.
- Mantén compatibilidad con Windows y Python 3.10/3.11.
- Agrega o actualiza pruebas para toda lógica nueva.
- Ejecuta `python -m pytest -q` antes de terminar.
- Reporta comandos ejecutados, archivos modificados, pruebas y pendientes.
- No presentes `spotted` como diagnóstico de hongo; es una categoría visual.
- Los datos sintéticos solo pueden incorporarse a `train` después de revisión humana.

## Comandos base

```powershell
pip install -r requirements-core.txt
python -m pytest -q
python scripts/check_environment.py
```

## Flujo Git

- No desarrollar directamente en `main`.
- Una rama por etapa: `feature/...`, `docs/...`, `fix/...`.
- Commits pequeños y descriptivos.
- Pull Request con instrucciones de prueba y resultados.
