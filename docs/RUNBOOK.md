# Runbook

## Entorno

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-core.txt
pip install -r requirements-app.txt
pip install -r requirements-vision.txt
pip install -r requirements-rag.txt
```

## Verificar entorno

```powershell
python scripts/check_environment.py
python -m compileall app src scripts
python -m pytest -q
```

## Comprobar artefactos

```powershell
Test-Path models\vision\resnet18_baseline_best.pt
Test-Path models\lora\soybean_sd15\pytorch_lora_weights.safetensors
Test-Path data\metadata\dataset_split.csv
Test-Path vector_db\index.faiss
Test-Path vector_db\metadata.json
Test-Path data\documents
```

## Ejecutar Streamlit

Smoke finito:

```powershell
python scripts/run_demo.py
```

Uso interactivo:

```powershell
python scripts/run_demo.py --serve --port 8501
```

Detener Streamlit interactivo: presionar `Ctrl+C` en la terminal.

## Smoke y prueba funcional

```powershell
python scripts/smoke_test_app.py
python scripts/run_functional_test.py
```

## Problemas comunes

- Si falta `models/vision/resnet18_baseline_best.pt`, la app inicia pero no podra clasificar hasta restaurar el checkpoint local.
- Si falta `vector_db/index.faiss` o el modelo de embeddings no esta cacheado, el pipeline usa metadata local cuando `vector_db/metadata.json` existe.
- Si faltan documentos reales y metadata, el RAG queda no disponible y el informe debe declarar la limitacion.
