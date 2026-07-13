# SeedCare-RAG LoRA

## Título final

**SeedCare-RAG LoRA: Sistema multimodal para clasificar defectos visibles en semillas de soja, recuperar evidencia técnica mediante RAG y ampliar experimentalmente los datos de entrenamiento con Stable Diffusion 1.5 ajustado con LoRA**

## Descripción puntual

El proyecto desarrolla una aplicación que recibe una fotografía de una semilla de soja y utiliza un modelo visual ajustado para clasificarla en una de cinco categorías observables: `intact`, `spotted`, `immature`, `broken` o `skin_damaged`. A partir de la predicción, un módulo RAG recupera información técnica sobre posibles causas, control de calidad, prevención, almacenamiento y manejo. Finalmente, el sistema genera un informe preliminar con fuentes, nivel de confianza y limitaciones.

Como evidencia de entrenamiento generativo, Stable Diffusion 1.5 se ajustará mediante LoRA con un subconjunto documentado del dataset. Las imágenes sintéticas serán revisadas y se usarán únicamente en el conjunto de entrenamiento para comparar el desempeño del clasificador con y sin datos sintéticos.

## Estado actual

- Repositorio y estructura: preparados.
- Dataset: **descargado**; cinco carpetas disponibles y estructura verificada.
- Auditoría completa del dataset: **pendiente de nueva ejecución** con los seis reportes esperados.
- Corpus documental del RAG: **aún no recopilado**.
- Entrenamientos: **pendientes**.
- Aplicación final: **pendiente de integración**.

## Dataset principal previsto

- Nombre: Soybean Seeds, versión 6.
- Fuente: Mendeley Data.
- DOI: `10.17632/v6vzvfszj6.6`.
- Total publicado: 5513 imágenes individuales.
- Clases: `intact`, `spotted`, `immature`, `broken`, `skin_damaged`.
- Licencia: CC BY 4.0.

La etiqueta `spotted` describe una anomalía visible; no confirma por sí sola hongos o una enfermedad específica.

## Entrenamientos que realizará el equipo

1. **Fine-tuning visual:** ResNet18 o EfficientNet-B0 para clasificar las cinco categorías.
2. **Stable Diffusion 1.5 + LoRA:** ajuste generativo para crear ejemplos sintéticos controlados.
3. **Comparación experimental:** clasificador con datos reales frente al clasificador con datos reales más imágenes sintéticas aceptadas.

El RAG no sustituye esos entrenamientos: recupera evidencia documental y fundamenta el informe generado.

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

## Preparacion del dataset LoRA SD 1.5

El objetivo del entrenamiento LoRA es ajustar Stable Diffusion 1.5 para generar imagenes de
semillas de soja condicionadas por la palabra trigger `soyseed` y por las cinco categorias
visuales originales. Esta etapa solo prepara el dataset; no entrena Stable Diffusion ni descarga
modelos.

La preparacion usa exclusivamente imagenes reales de `data/processed/train/`. No utiliza
`data/processed/validation/` ni `data/processed/test/`.

Validar candidatos sin crear archivos:

```powershell
python scripts/prepare_lora_dataset.py --config configs/lora_sd15_config.yaml --dry-run
```

Preparar el dataset LoRA:

```powershell
python scripts/prepare_lora_dataset.py --config configs/lora_sd15_config.yaml --overwrite
```

Validar el dataset preparado:

```powershell
python scripts/validate_lora_dataset.py --config configs/lora_sd15_config.yaml
```

Salidas esperadas:

- Imagenes copiadas y normalizadas: `data/lora/train/images/`
- Metadata de captions: `data/lora/train/metadata.jsonl`
- Reporte de seleccion: `data/lora/train/selection_report.csv`

Las imagenes preparadas, metadata completa, pesos LoRA y logs de entrenamiento no deben subirse a
GitHub. El reporte de seleccion puede versionarse cuando contenga rutas relativas y no rutas
privadas absolutas.

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
python -m pytest -q
python scripts/check_environment.py
```

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
