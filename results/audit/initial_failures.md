# Fallos iniciales

## Error original reproducido

Comando:

```powershell
python app\app.py
```

Resultado:

```text
ModuleNotFoundError: No module named 'app.components'; 'app' is not a package
```

Causa inicial confirmada: `app/app.py` se ejecuta como archivo de script. En ese modo, Python resuelve el nombre `app` contra el propio archivo `app.py`, no contra el paquete `app/`, y la importación absoluta `from app.components.demo_helpers import ...` falla.

## Fallo de cobertura del smoke test actual

Comando:

```powershell
python scripts/run_demo.py
```

Resultado inicial:

```text
Streamlit inicio correctamente en http://localhost:<puerto>
```

Ese resultado solo valida `/_stcore/health`. No garantiza que la sesión de Streamlit haya ejecutado correctamente el script ni que la página funcional responda sin traceback.

## Riesgos secundarios detectados

- `scripts/run_demo.py` apunta a `app/app.py`, el nombre que causa la colisión al ejecutarse como script.
- La app requiere FAISS y embeddings al ejecutar el análisis; debe validar y comunicar honestamente el estado si esos artefactos no están disponibles.
- `requirements.txt` instala solo dependencias core; para la app se requiere `requirements-app.txt` y para visión/RAG se requieren archivos adicionales.
- `scripts/evaluate_system.py` todavía es un placeholder y no forma parte del flujo funcional validado.
- Documentación existente aún referencia `app/app.py` y estados pendientes.

## Estado inicial de pruebas

- `python scripts/check_environment.py`: pasa.
- `python -m pytest -q`: 48 pruebas pasan.
- `python scripts/run_demo.py`: healthcheck pasa, pero es insuficiente.
- `python app\app.py`: falla con el error original.
