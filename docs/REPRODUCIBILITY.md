# Reproducibilidad

## Entorno

Requisitos recomendados:

- Windows con PowerShell.
- Python 3.10 o 3.11.
- Entorno virtual local `.venv`.

Instalacion:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-core.txt
pip install -r requirements-vision.txt
pip install -r requirements-rag.txt
pip install -r requirements-app.txt
```

Verificacion:

```powershell
python scripts/check_environment.py
python -m pytest -q
```

## Datos

Dataset principal registrado: Soybean Seeds version 6, DOI `10.17632/v6vzvfszj6.6`.

Reglas:

- No modificar `data/raw/`.
- Copiar datos procesados a `data/processed/`.
- Mantener `validation` y `test` solo con imagenes reales.
- No versionar datasets completos.

Preparacion:

```powershell
python scripts/verify_dataset_structure.py --dataset data/raw/soybean_seeds
python scripts/audit_dataset.py --dataset data/raw/soybean_seeds --output results/dataset_audit
python scripts/prepare_dataset.py `
  --dataset data/raw/soybean_seeds `
  --audit-dir results/dataset_audit `
  --output data/processed `
  --metadata-output data/metadata `
  --seed 42
python scripts/validate_dataset_split.py --dataset-split data/metadata/dataset_split.csv
```

## Vision

Entrenamiento:

```powershell
python scripts/train_vision_model.py --config configs/vision_config.yaml
```

Evaluacion:

```powershell
python scripts/evaluate_vision_model.py `
  --config configs/vision_config.yaml `
  --checkpoint models/vision/resnet18_baseline_best.pt
```

Resultado canonico actual: `results/vision/resnet18_baseline/metrics_test.json`.

## RAG

Preparacion de corpus:

```powershell
python scripts/prepare_rag_corpus.py `
  --input data/documents/inbox `
  --accepted data/documents/accepted `
  --rejected data/documents/rejected `
  --metadata data/metadata/document_sources.csv `
  --results results/rag
```

Construccion de indice:

```powershell
python scripts/build_vector_db.py `
  --config configs/rag.yaml `
  --documents data/documents/accepted `
  --sources data/metadata/document_sources.csv `
  --output vector_db
```

Evaluacion:

```powershell
python scripts/evaluate_rag.py `
  --config configs/rag.yaml `
  --index vector_db `
  --queries data/metadata/rag_evaluation_queries.csv `
  --output results/rag/evaluation
```

## Demo

Streamlit:

```powershell
python scripts/run_demo.py --serve
```

CLI:

```powershell
python scripts/analyze_seed.py `
  --image data/processed/validation/immature/1.jpg `
  --output results/reports/demo_cli_report.json `
  --device cpu
```

## LoRA

Configuracion registrada en `configs/lora_sd15.yaml`. Evidencia consolidada en `results/lora/`.

No se deben versionar pesos `.safetensors`. La evidencia actual confirma adaptador local y parametros, pero no confirma hardware ni tiempo de entrenamiento.

## Artefactos no versionables

- `data/raw/`
- datasets completos procesados cuando sean pesados o privados;
- `models/**/*.pt`;
- `models/**/*.safetensors`;
- `vector_db/index.faiss`;
- caches;
- tokens;
- `.env`.

## Reproducibilidad de resultados finales

La lectura final debe partir de:

- `results/system/final_metrics.json`;
- `results/vision/resnet18_baseline/reconciliation_report.md`;
- `results/rag/evaluation/evaluation_report.md`;
- `results/lora/evidence_report.md`;
- `results/dataset_preparation/summary.json`.
