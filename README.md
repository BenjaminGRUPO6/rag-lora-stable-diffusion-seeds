# SeedCare-RAG LoRA

## Título final

**SeedCare-RAG LoRA: Sistema multimodal para clasificar defectos visibles en semillas de soja, recuperar evidencia técnica mediante RAG y ampliar experimentalmente los datos de entrenamiento con Stable Diffusion 1.5 ajustado con LoRA**

## Descripción puntual

El proyecto desarrolla una aplicación que recibe una fotografía de una semilla de soja y utiliza un modelo visual ajustado para clasificarla en una de cinco categorías observables: `intact`, `spotted`, `immature`, `broken` o `skin_damaged`. A partir de la predicción, un módulo RAG recupera información técnica sobre posibles causas, control de calidad, prevención, almacenamiento y manejo. Finalmente, el sistema genera un informe preliminar con fuentes, nivel de confianza y limitaciones.

Como evidencia de entrenamiento generativo, Stable Diffusion 1.5 se ajustará mediante LoRA con un subconjunto documentado del dataset. Las imágenes sintéticas serán revisadas y se usarán únicamente en el conjunto de entrenamiento para comparar el desempeño del clasificador con y sin datos sintéticos.

## Estado actual

- Repositorio y estructura: preparados.
- Dataset: **completado**.
- Baseline ResNet18: **entrenado**; metricas por reconciliar.
- Stable Diffusion 1.5 + LoRA: **entrenado**; evidencia por consolidar.
- Experimento B con imagenes sinteticas en ResNet18: **aplazado** como trabajo futuro.
- Corpus documental e indice del RAG: **disponibles localmente**.
- Aplicación Streamlit: **integrada y auditada funcionalmente**.

## Dataset principal previsto

- Nombre: Soybean Seeds, versión 6.
- Fuente: Mendeley Data.
- DOI: `10.17632/v6vzvfszj6.6`.
- Total publicado: 5513 imágenes individuales.
- Clases: `intact`, `spotted`, `immature`, `broken`, `skin_damaged`.
- Licencia: CC BY 4.0.

La etiqueta `spotted` describe una anomalía visible; no confirma por sí sola hongos o una enfermedad específica.

## Entrenamientos y evaluacion

1. **Fine-tuning visual:** baseline ResNet18 para clasificar las cinco categorias.
2. **Stable Diffusion 1.5 + LoRA:** ajuste generativo para evaluar el comportamiento del LoRA entrenado.
3. **Trabajo futuro:** segundo entrenamiento de ResNet18 con datos sinteticos aceptados despues de revision humana.

El RAG no sustituye esos entrenamientos: recupera evidencia documental y fundamenta el informe generado.

## Evidencia local del entrenamiento LoRA SD1.5

El ajuste Stable Diffusion 1.5 + LoRA ya fue realizado y la etapa actual solo consolida evidencia
reproducible desde artefactos locales. No se ejecuto reentrenamiento ni inferencia masiva durante
la consolidacion.

Artefactos locales usados como evidencia:

- Configuracion: `configs/lora_sd15.yaml`
- Metadata de entrenamiento: `data/lora/train/metadata.jsonl`
- Notebook de entrenamiento: `notebooks/06_entrenamiento_lora_sd15_colab.ipynb`
- Pesos locales del adaptador: `models/lora/soybean_sd15/pytorch_lora_weights.safetensors`
- Evidencia consolidada: `results/lora/`

Los pesos LoRA son artefactos locales y no deben versionarse en Git. Los reportes consolidados
registran la evidencia disponible, los faltantes y el estado `PARTIAL` cuando no existan salidas
ejecutadas del notebook, logs, hardware, tiempo de entrenamiento o comparativas base vs. LoRA.

## Experimento A: ResNet18 con imagenes reales

El Experimento A ajusta un clasificador ResNet18 preentrenado usando solo imagenes reales de
`data/processed/train/`. La seleccion del mejor checkpoint se hace con
`data/processed/validation/` mediante macro-F1. El split `data/processed/test/` se usa una sola
vez al final para reportar el desempeno definitivo.

Clases utilizadas, en orden:

1. `intact`
2. `spotted`
3. `immature`
4. `broken`
5. `skin_damaged`

Smoke test en CPU:

```powershell
python scripts/train_vision_model.py --config configs/vision_config.yaml --smoke-test --device cpu
```

Entrenamiento real:

```powershell
python scripts/train_vision_model.py --config configs/vision_config.yaml
```

Evaluacion final del checkpoint:

```powershell
python scripts/evaluate_vision_model.py `
  --config configs/vision_config.yaml `
  --checkpoint models/vision/resnet18_baseline_best.pt
```

Resultados esperados:

- Checkpoint: `models/vision/resnet18_baseline_best.pt`
- Metricas, CSV y graficos: `results/vision/resnet18_baseline/`

Los checkpoints no deben subirse a Git. Las metricas, CSV y graficos seleccionados del
entrenamiento real pueden versionarse si son necesarios para documentar el experimento.

## Arquitectura

```text
Imagen de semilla
      ↓
Clasificador visual ajustado
      ↓
Categoría + confianza
      ↓
Consulta automática al RAG
      ↓
Recuperación de documentos y fragmentos
      ↓
Informe preliminar con fuentes y limitaciones
```

Flujo experimental de LoRA:

```text
Imágenes reales + captions
      ↓
Stable Diffusion 1.5 ajustado con LoRA
      ↓
Imágenes sintéticas pendientes de revisión
      ↓
Selección humana
      ↓
Entrenamiento comparativo del clasificador
```

## Inicio rápido

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-core.txt
pip install -r requirements-app.txt
pip install -r requirements-vision.txt
pip install -r requirements-rag.txt
python -m pytest -q
python scripts/check_environment.py
python scripts/run_demo.py
```

La demo oficial queda activa hasta presionar `Ctrl+C`:

```powershell
python scripts/run_demo.py --port 8501
```

Tambien puede iniciarse directamente con Streamlit, sin configurar `PYTHONPATH`:

```powershell
python -m streamlit run app/streamlit_app.py --server.port 8501
```

El unico entrypoint Streamlit mantenido es `app/streamlit_app.py`. El antiguo
`app/app.py` fue retirado porque al ejecutarse como script sombreaba el paquete
`app/` y podia provocar `ModuleNotFoundError`. El entrypoint actual inserta la
raiz del repositorio en `sys.path` antes de importar `app.*` o `src.*`, una
solucion estable en Windows tanto desde la raiz como desde otro directorio.

## Primera etapa

1. Mantener el dataset fuera de Git en `data/raw/soybean_seeds/`.
2. Confirmar que existen las cinco carpetas oficiales.
3. Ejecutar `verify_dataset_structure.py` cuando cambie la estructura local.
4. Ejecutar la auditoría completa antes de dividir o aumentar datos.
5. Revisar los seis reportes de auditoría antes de continuar con splits.

```powershell
python scripts/verify_dataset_structure.py --dataset data/raw/soybean_seeds
python scripts/audit_dataset.py --dataset data/raw/soybean_seeds --output results/dataset_audit
```

## Documentación

La carpeta `docs/` contiene el contexto completo, metodología, plan de entrenamiento, GitHub, Codex, evaluación, ética y estructura del informe IEEE.

## Depuracion controlada y split

Politica de depuracion:

- `data/raw/` es inmutable: no eliminar, renombrar, mover ni modificar archivos originales.
- La preparacion copia imagenes validas hacia `data/processed/`; nunca mueve datos.
- Los archivos corruptos se excluyen.
- Los duplicados exactos se agrupan por `sha256`; se conserva una sola imagen canonica por grupo y se registra la razon de exclusion.
- Los posibles duplicados visuales no se excluyen automaticamente. Se registran en `data/metadata/near_duplicates_review.csv` para revision humana.
- Si un par visual se marca como equivalente en la revision, sus imagenes deben quedar en el mismo split.
- Las imagenes sinteticas solo pueden entrar en `train` despues de revision humana; nunca en `validation` ni `test`.

Preparar el dataset limpio y dividido:

```powershell
python scripts/prepare_dataset.py `
  --dataset data/raw/soybean_seeds `
  --audit-dir results/dataset_audit `
  --output data/processed `
  --metadata-output data/metadata `
  --seed 42
```

Validar el split generado:

```powershell
python scripts/validate_dataset_split.py --dataset-split data/metadata/dataset_split.csv
```

Archivos generados por la preparacion:

- `data/metadata/exclusions.csv`
- `data/metadata/near_duplicates_review.csv`
- `data/metadata/dataset_split.csv`
- `results/dataset_preparation/summary.json`
- `results/dataset_preparation/class_distribution_by_split.csv`
