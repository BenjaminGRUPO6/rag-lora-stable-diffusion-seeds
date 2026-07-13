# Functional Audit

## Estado inicial

Error original reproducido con:

```powershell
python app\app.py
```

Resultado:

```text
ModuleNotFoundError: No module named 'app.components'; 'app' is not a package
```

## Causa raiz

`app/app.py` se ejecutaba como script y ocultaba el paquete `app/`. Por eso la importacion absoluta `from app.components.demo_helpers import ...` resolvia `app` como modulo de archivo y no como paquete.

## Errores secundarios

- `scripts/run_demo.py` apuntaba al entrypoint conflictivo `app/app.py`.
- El smoke test previo solo validaba `/_stcore/health`, no el render real de la app.
- Streamlit/AppTest podia ejecutar el entrypoint sin la raiz del repo en `sys.path`.
- El RAG intentaba cargar SentenceTransformer con acceso de red si el modelo no estaba cacheado.
- La app no tenia un fallback local explicito cuando FAISS/embeddings no estaban disponibles.

## Cambios aplicados

- Renombrado el entrypoint a `app/streamlit_app.py`.
- `scripts/run_demo.py` usa rutas absolutas calculadas desde `__file__` y ejecuta Streamlit con `cwd` en la raiz del repo.
- `app/streamlit_app.py` inserta la raiz del repo de forma controlada antes de importar `app.*` y `src.*`.
- `TextEmbedder` usa `local_files_only=True` por defecto.
- `src.pipelines.analyze_seed` incorpora fallback RAG local por metadata cuando FAISS/embeddings no estan disponibles.
- Se agrego `scripts/smoke_test_app.py`.
- Se agrego `scripts/run_functional_test.py`.
- Se agregaron pruebas de imports, startup, RAG fallback, checkpoint, smoke helpers y prueba funcional.

## Ciclos

Total de ciclos diagnosticar/corregir/probar: 6.

## Pruebas ejecutadas

```powershell
python scripts/check_environment.py
python -m compileall app src scripts
python -m pytest -q
python scripts/run_demo.py
python scripts/smoke_test_app.py
python scripts/run_functional_test.py
```

## Resultados

- `python -m pytest -q`: 61 passed.
- `python -m compileall app src scripts`: sin errores.
- `python scripts/check_environment.py`: entorno core correcto.
- `python scripts/smoke_test_app.py`: PASS, HTTP 200, puerto activo, titulo renderizado.
- `python scripts/run_functional_test.py`: passed true.

## Artefactos

- `models/vision/resnet18_baseline_best.pt`: AVAILABLE.
- `models/lora/soybean_sd15/pytorch_lora_weights.safetensors`: AVAILABLE.
- `data/metadata/dataset_split.csv`: AVAILABLE.
- `vector_db/index.faiss`: AVAILABLE.
- `vector_db/metadata.json`: AVAILABLE.
- `data/documents/`: AVAILABLE.
- `results/vision/resnet18_baseline/`: AVAILABLE.

## Limitaciones

- El RAG depende de la calidad de los chunks ya indexados; algunos fragmentos PDF contienen texto extraido con ruido.
- Si el modelo SentenceTransformer no esta cacheado, el sistema usa recuperacion lexical local sobre `vector_db/metadata.json`.
- No se entreno ResNet18, Stable Diffusion ni LoRA durante esta auditoria.

## Estado final

SUCCESS.
