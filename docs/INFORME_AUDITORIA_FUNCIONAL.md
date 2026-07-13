# Informe de Auditoria Funcional

## Portada

Proyecto: SeedCare-RAG LoRA  
Repositorio: `rag-lora-stable-diffusion-seeds`  
Fecha: 2026-07-13  
Entorno: Windows 10, Python 3.10.11

## Estado inicial

La aplicacion Streamlit presentaba:

```text
ModuleNotFoundError: No module named 'app.components'; 'app' is not a package
```

El error se reprodujo con:

```powershell
python app\app.py
```

## Diagnostico y causa raiz

El archivo `app/app.py` ocultaba el paquete `app/` cuando se ejecutaba como script. Python resolvia `app` como el archivo `app.py`, no como el paquete, y fallaba la importacion `app.components.demo_helpers`.

Se detectaron causas secundarias:

- `scripts/run_demo.py` apuntaba al entrypoint conflictivo.
- El smoke test anterior no validaba el render real.
- El RAG podia intentar red para cargar SentenceTransformer.
- Faltaba fallback local cuando embeddings no estaban disponibles.

En la revision final de 2026-07-13 se confirmo que `app/app.py` ya no existe como fuente, `streamlit_app.py` no importa ni ejecuta `app.app`, y el artefacto compilado obsoleto `app/__pycache__/app.cpython-310.pyc` fue eliminado.

## Cambios realizados

- Entry point final: `app/streamlit_app.py`.
- Comando oficial: `python scripts/run_demo.py --port 8501`.
- Comando directo equivalente: `python -m streamlit run app/streamlit_app.py --server.port 8501`.
- Rutas absolutas desde `__file__` para independencia del cwd.
- `app/streamlit_app.py` inserta la raiz del repositorio en `sys.path` antes de importar `app.*` o `src.*`.
- Fallback local `MetadataKeywordRetriever` sobre `vector_db/metadata.json`.
- `TextEmbedder(local_files_only=True)` por defecto.
- Smoke test real en `scripts/smoke_test_app.py`.
- Prueba funcional en `scripts/run_functional_test.py`.
- `scripts/smoke_test_app.py` falla ante `ModuleNotFoundError`, `No module named 'app'`, `Traceback` o `ImportError` en stdout/stderr.

## Arquitectura final

```text
Imagen local o cargada
  -> validacion PIL
  -> ResNet18 local
  -> probabilidades
  -> RAG FAISS local o fallback metadata
  -> informe preliminar determinista
  -> limitaciones y fuentes
```

## Archivos creados y modificados

Archivos creados principales:

- `app/streamlit_app.py`
- `scripts/smoke_test_app.py`
- `scripts/run_functional_test.py`
- `tests/test_app_startup.py`
- `tests/test_smoke_test_app.py`
- `tests/test_vision_inference.py`
- `tests/test_rag_embeddings.py`
- `tests/test_functional_test_script.py`
- `docs/FUNCTIONAL_AUDIT.md`
- `docs/CHANGES_IMPLEMENTED.md`
- `docs/RUNBOOK.md`
- `docs/KNOWN_LIMITATIONS.md`
- `docs/RELEASE_CHECKLIST.md`

Archivos modificados principales:

- `scripts/run_demo.py`
- `src/pipelines/analyze_seed.py`
- `src/rag/retrieval.py`
- `src/rag/embeddings.py`
- `README.md`
- `PROJECT_STATUS.md`

## Pruebas ejecutadas

```powershell
python scripts/check_environment.py
python -m compileall app src scripts
python -m pytest -q
python -m streamlit run app/streamlit_app.py --server.port 8501
python scripts/run_demo.py --port 8501
python scripts/smoke_test_app.py
python scripts/run_functional_test.py
```

## Resultados de pytest

```text
65 passed
```

## Resultado del smoke test

Archivo: `results/app_smoke_test/summary.json`

- Estado: PASS.
- HTTP status: 200.
- Puerto escuchando: true.
- Proceso vivo durante la prueba: true.
- Titulo renderizado: true.
- Streamlit exceptions: 0.
- Forbidden log found: false.

La prueba temporal directa en puerto 8501 confirmo proceso activo, puerto escuchando, HTTP 200, titulo renderizado por `AppTest`, cero excepciones y ausencia de `ModuleNotFoundError`, `No module named 'app'`, `Traceback` e `ImportError`.

## Resultado de la prueba funcional

Archivo: `results/end_to_end/functional_test.json`

- passed: true.
- Imagen: `data/processed/test/broken/10.jpg`.
- Checkpoint disponible: true.
- Prediccion: `skin_damaged`.
- Confianza: 0.4687288701534271.
- Suma de probabilidades: 0.9999999664723873.
- RAG status: `faiss`.
- Fuentes recuperadas: 5.

## Artefactos encontrados

- `models/vision/resnet18_baseline_best.pt`: disponible.
- `models/lora/soybean_sd15/pytorch_lora_weights.safetensors`: disponible.
- `data/metadata/dataset_split.csv`: disponible.
- `vector_db/index.faiss`: disponible.
- `vector_db/metadata.json`: disponible.
- `data/documents/`: disponible.
- `results/vision/resnet18_baseline/`: disponible.

## Limitaciones

- El informe es preliminar y no es diagnostico fitosanitario.
- `spotted` se trata solo como categoria visual.
- El fallback lexical local es una degradacion honesta si embeddings no estan disponibles.
- Algunos chunks PDF contienen ruido de extraccion.

## Riesgos

- Si se borra el checkpoint local, la clasificacion no podra ejecutarse hasta restaurarlo.
- Si se borra `vector_db/metadata.json`, el fallback RAG no tendra corpus local.
- Los pesos y datasets no deben agregarse a Git.

## Instrucciones de ejecucion

```powershell
pip install -r requirements-core.txt
pip install -r requirements-app.txt
pip install -r requirements-vision.txt
pip install -r requirements-rag.txt
python -m pytest -q
python scripts/check_environment.py
python scripts/run_demo.py --port 8501
python -m streamlit run app/streamlit_app.py --server.port 8501
```

## Conclusion

Estado final: SUCCESS. La aplicacion inicia, responde HTTP 200, renderiza el titulo esperado, ejecuta clasificacion con checkpoint local, recupera fuentes reales con RAG local y genera una salida estructurada valida sin entrenar modelos ni inventar fuentes.
