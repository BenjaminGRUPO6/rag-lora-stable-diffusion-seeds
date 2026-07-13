# Contexto actual del proyecto

Documento generado a partir de la inspeccion local del repositorio. No incluye resultados inferidos fuera de los archivos presentes.

## 1. Nombre y objetivo actual del proyecto

- Nombre del paquete: `rag-lora-stable-diffusion-seeds`.
- Nombre del proyecto en README/configuracion: `SeedCare-RAG LoRA`.
- Objetivo actual declarado: clasificar defectos visibles en semillas de soja en cinco categorias (`intact`, `spotted`, `immature`, `broken`, `skin_damaged`), recuperar evidencia tecnica mediante RAG y experimentar con ampliacion sintetica usando Stable Diffusion 1.5 + LoRA.
- Nota de alcance: `spotted` aparece documentado como categoria visual; no confirma hongos ni enfermedad especifica.

## 2. Rama actual

- Rama activa: `feature/experiment-b-intact-broken`.
- Estado Git encontrado:
  - `models/README.md` modificado.
  - `configs/synthetic_generation.yaml` sin rastrear.
  - `data/lora/` sin rastrear.
- La rama local aparece asociada a `origin/feature/experiment-b-intact-broken`.

## 3. Ultimos 15 commits

```text
2994e01 (HEAD -> feature/experiment-b-intact-broken, origin/feature/experiment-b-intact-broken, main) Merge branch 'main' of https://github.com/BenjaminGRUPO6/rag-lora-stable-diffusion-seeds
0aa7a97 (origin/main, origin/HEAD) Merge pull request #5 from BenjaminGRUPO6/feature/resnet18-baseline-training
5547181 (origin/feature/resnet18-baseline-training, feature/resnet18-baseline-training) feat: entrena baseline ResNet18 para defectos visibles en semillas
c53227e feat: implementa entrenamiento y evaluacion baseline ResNet18
176d80d Merge branch 'feature/dataset-cleaning-split'
26d8586 (origin/feature/dataset-cleaning-split, feature/dataset-cleaning-split) feat: depura y divide el dataset Soybean Seeds sin fuga de datos
a347d01 Merge pull request #4 from BenjaminGRUPO6/feature/dataset-cleaning-split
844698d feat: agrega limpieza y division segura del dataset
df53f7f Merge pull request #3 from BenjaminGRUPO6/feature/dataset-soybean-seeds
fd8b40d (origin/feature/dataset-soybean-seeds, feature/dataset-soybean-seeds) fix: corrige coherencia y consolida configuraciones del proyecto
9121e12 fix: completa auditoria y mantiene compatibilidad
283888c fix: mantiene compatibilidad y completa auditoria del dataset
337963e Merge pull request #2 from BenjaminGRUPO6/feature/dataset-soybean-seeds
cbf5899 Implementar auditoria y verificacion del dataset de semillas
ddca25f Merge pull request #1 from BenjaminGRUPO6/feature/dataset-audit
```

## 4. Estructura relevante del repositorio

```text
app/                     Interfaz Streamlit minima.
codex/                   Planes y tareas por etapa.
configs/                 Configuracion de dataset, vision, RAG, LoRA y generacion sintetica.
data/                    Datos locales, metadatos y plantillas; data/raw y data/processed no se deben versionar.
docs/                    Documentacion del proyecto e informe.
models/                  README y pesos locales no versionables.
notebooks/               Notebooks por etapa, sin salidas ejecutadas.
results/                 Resultados de auditoria, preparacion, vision y placeholders de otros modulos.
scripts/                 Entrypoints CLI.
src/data/                Auditoria, limpieza, verificacion y split.
src/vision/              Dataset, modelo, entrenamiento, inferencia y evaluacion ResNet18.
src/synthetic_data/      Utilidades para captions, comando LoRA, generacion y revision.
src/rag/                 Chunking, carga documental, embeddings, FAISS, retrieval y prompt.
src/pipelines/           Integracion parcial de RAG e informe.
tests/                   Pruebas unitarias del estado implementado.
vector_db/               README; indice regenerable no versionado.
```

## 5. Etapas implementadas

- Estructura base del repositorio, configuraciones y documentacion inicial.
- Verificacion de estructura del dataset.
- Auditoria del dataset: existen reportes en `results/dataset_audit/`.
- Limpieza y split sin fuga por hash exacto: existen `data/metadata/dataset_split.csv`, `data/metadata/exclusions.csv`, `data/metadata/near_duplicates_review.csv` y resumen en `results/dataset_preparation/`.
- Entrenamiento y evaluacion de baseline ResNet18: existe checkpoint local y resultados en `results/vision/resnet18_baseline/`.
- Pruebas unitarias para dataset, vision, RAG basico, reportes, prompts y comando LoRA.
- App Streamlit inicial, sin integracion real del clasificador ni RAG.
- Modulos RAG basicos implementados, pero sin corpus tecnico poblado ni indice usable versionado.
- Utilidades de datos sinteticos implementadas parcialmente en `src/synthetic_data/`.

## 6. Etapas pendientes

- Resolver el estado Git no limpio, especialmente `data/lora/` sin rastrear y el `models/README.md` modificado.
- Confirmar si `configs/synthetic_generation.yaml` debe versionarse como parte del Experimento B.
- Completar o reemplazar `scripts/prepare_lora_dataset.py`, que sigue como placeholder aunque existe logica en `src/synthetic_data/prepare_lora_dataset.py`.
- Reunir y registrar corpus tecnico en `data/documents/` y `data/metadata/document_sources.csv`.
- Construir indice RAG con un script ejecutable real; `scripts/build_vector_db.py` aun es placeholder.
- Integrar Streamlit con clasificador entrenado, RAG e informe.
- Revisar humanamente imagenes sinteticas antes de incorporarlas a `train`.
- Ejecutar y documentar el Experimento B con comparacion controlada; no hay metricas existentes de ese experimento.
- Consolidar metricas finales del sistema; `scripts/evaluate_system.py` solo crea carpeta y muestra una instruccion.

## 7. Scripts disponibles y para que sirven

- `scripts/check_environment.py`: verifica paquetes core (`PIL`, `yaml`, `pandas`, `numpy`, `sklearn`, `imagehash`, `pytest`) y version de Python.
- `scripts/verify_dataset_structure.py`: valida carpetas esperadas del dataset y cuenta archivos por categoria.
- `scripts/audit_dataset.py`: audita imagenes, corruptos, tamanos, duplicados exactos y posibles duplicados visuales; escribe seis reportes.
- `scripts/prepare_dataset.py`: copia imagenes validas a `data/processed/`, genera metadatos y split `train/validation/test`; contiene al inicio un bloque antiguo desactivado con mensaje de pendiente.
- `scripts/validate_dataset_split.py`: valida columnas, clases, duplicados, fugas por hash, sinteticos en holdout, proporciones y existencia de rutas del split.
- `scripts/train_vision_model.py`: entrena el Experimento A ResNet18 con opcion `--smoke-test`, overrides y `--resume`.
- `scripts/evaluate_vision_model.py`: evalua un checkpoint ResNet18 sobre test y guarda metricas/predicciones.
- `scripts/prepare_lora_dataset.py`: placeholder que imprime que la preparacion LoRA esta pendiente.
- `scripts/build_vector_db.py`: placeholder que imprime que falta construir la base vectorial tras reunir documentos.
- `scripts/evaluate_system.py`: crea `results/metrics` y pide completar matrices de evaluacion; no consolida metricas reales.
- `scripts/run_demo.py`: ejecuta `streamlit run app/streamlit_app.py` desde la raiz del repositorio.

## 8. Notebooks disponibles y salidas ejecutadas

Todos los notebooks inspeccionados tienen `cells_with_outputs=0` y `executed_code_cells=0`.

- `00_configuracion_entorno.ipynb`
- `01_adquisicion_y_auditoria.ipynb`
- `01_auditoria_dataset.ipynb`
- `02_limpieza_y_etiquetado.ipynb`
- `02_splits_y_aumento.ipynb`
- `03_division_y_aumento.ipynb`
- `03_entrenamiento_visual.ipynb`
- `04_entrenamiento_modelo_visual.ipynb`
- `04_preparacion_lora.ipynb`
- `05_entrenamiento_lora_colab.ipynb`
- `05_preparacion_lora_sd15.ipynb`
- `06_entrenamiento_lora_sd15_colab.ipynb`
- `06_revision_sinteticos.ipynb`
- `07_construccion_rag.ipynb`
- `08_integracion_informe.ipynb`
- `08_integracion_y_evaluacion.ipynb`
- `09_evaluacion_sistema.ipynb`

## 9. Modelos o checkpoints esperados, sin copiar pesos

- Esperado por configuracion de vision: `models/vision/resnet18_baseline_best.pt`.
- Encontrado localmente: `models/vision/resnet18_baseline_best.pt` (peso local, no copiar ni subir).
- Encontrado smoke test local: `results/vision/smoke_test/smoke_test_best.pt` (checkpoint local regenerable).
- Esperado por `configs/lora_sd15.yaml`: salida LoRA en `models/lora`.
- Encontrado localmente: `models/lora/soybean_sd15/pytorch_lora_weights.safetensors` (peso local, no copiar ni subir).
- `checkpoints/` solo contiene `.gitkeep`.
- `models/README.md` indica que pesos/checkpoints no se versionan, pero el archivo esta modificado y contiene lineas de patron tipo `.gitignore`.

## 10. Resultados y metricas existentes

### Dataset

- `results/dataset_audit/summary.json`:
  - `total_files`: 5513.
  - `valid_images`: 5513.
  - `corrupted_images`: 0.
  - `small_images`: 0.
  - `exact_duplicate_groups`: 290.
  - `exact_duplicate_files`: 580.
  - `possible_near_duplicate_pairs`: 123.
  - Distribucion: `broken` 1002, `immature` 1125, `intact` 1201, `skin_damaged` 1127, `spotted` 1058.
- `results/dataset_preparation/summary.json`:
  - `audit_total_files`: 5513.
  - `included_images`: 5223.
  - `excluded_images`: 290.
  - `exclusion_reasons`: `exact_duplicate` 290.
  - `split_counts`: train 4179, validation 522, test 522.
  - `synthetic_train_images`: 0.

### Vision ResNet18

- `results/vision/resnet18_baseline/run_summary.json`:
  - `best_validation_macro_f1`: 0.9231883922779236.
  - `test_macro_f1`: 0.9178406787519563.
  - `device`: `cuda`.
  - `epochs_ran`: 11.
  - `smoke_test`: false.
- `results/vision/resnet18_baseline/metrics_validation.json`:
  - `accuracy`: 0.9233716475095786.
  - `macro_f1`: 0.9231883922779236.
- `results/vision/resnet18_baseline/metrics_test.json`:
  - `accuracy`: 0.6704980842911877.
  - `macro_f1`: 0.6259550750897566.
  - Clases con menor F1 en test: `intact` 0.2962962962962963, `broken` 0.6419753086419753.
- Observacion real encontrada: hay inconsistencia entre `run_summary.json` (`test_macro_f1` 0.9178406787519563) y `metrics_test.json`/`classification_report.csv` (`macro_f1` 0.6259550750897566).
- Existen graficos PNG de curvas y matrices de confusion en `results/vision/resnet18_baseline/`; no se copian en este documento.

### Otros resultados

- `results/rag/index/` solo contiene `.gitkeep`.
- `results/synthetic/full/` solo contiene `.gitkeep`.
- No se encontro carpeta `results/lora/`.

## 11. Estado del dataset

- Dataset principal declarado: Soybean Seeds version 6, DOI `10.17632/v6vzvfszj6.6`, licencia CC BY 4.0.
- `data/raw/soybean_seeds` existe localmente con cinco categorias y conteos:
  - `broken`: 1002.
  - `immature`: 1125.
  - `intact`: 1201.
  - `skin_damaged`: 1127.
  - `spotted`: 1058.
- `data/processed/` existe con split preparado:
  - test: `broken` 100, `immature` 112, `intact` 91, `skin_damaged` 113, `spotted` 106.
  - train: `broken` 802, `immature` 898, `intact` 732, `skin_damaged` 901, `spotted` 846.
  - validation: `broken` 100, `immature` 112, `intact` 91, `skin_damaged` 113, `spotted` 106.
- `data/metadata/dataset_sources.csv` aun dice que la auditoria esta pendiente de nueva ejecucion completa, aunque existen reportes reales en `results/dataset_audit/`.
- `data/synthetic/pending_review/` y `data/synthetic/reviewed/` no contienen imagenes, solo `.gitkeep`.
- `data/lora/train/` existe sin rastrear con `metadata.jsonl` de 1000 lineas y 1000 imagenes: 200 por cada categoria.

## 12. Estado del entrenamiento ResNet18

- Implementado en `src/vision/` y ejecutable mediante `scripts/train_vision_model.py`.
- Configuracion activa principal: `configs/vision_config.yaml`.
- Checkpoint local encontrado: `models/vision/resnet18_baseline_best.pt`.
- Resultados reales presentes en `results/vision/resnet18_baseline/`.
- La validacion alcanzo `macro_f1` 0.9231883922779236 en los archivos existentes.
- El test registrado en `metrics_test.json` tiene `macro_f1` 0.6259550750897566 y `accuracy` 0.6704980842911877.
- Existe un smoke test en `results/vision/smoke_test/` con `test_macro_f1` 0.1; es evidencia de prueba tecnica, no de desempeno final.

## 13. Estado del entrenamiento Stable Diffusion 1.5 + LoRA

- Configuracion base en `configs/lora_sd15.yaml`.
- Utilidad para construir comando oficial Diffusers en `src/synthetic_data/train_sd15_lora.py`.
- Peso LoRA local encontrado: `models/lora/soybean_sd15/pytorch_lora_weights.safetensors`.
- No se encontraron metricas de entrenamiento LoRA ni carpeta `results/lora/`.
- `scripts/prepare_lora_dataset.py` esta pendiente, pero existe un dataset LoRA local sin rastrear en `data/lora/train/`.
- `data/lora/train/metadata.jsonl` usa rutas `images/...` y prompts con trigger `soyseed`.

## 14. Estado del Experimento B

- Existe configuracion sin rastrear: `configs/synthetic_generation.yaml`.
- La configuracion define comparacion para clases `intact` y `broken`, semillas `[42, 123, 456, 789, 1024]`, `prompts_per_class: 4`, `candidates_per_class: 150` y `target_accepted_per_class: 80`.
- Rutas esperadas por la configuracion:
  - `results/lora/base_comparison`
  - `results/lora/lora_comparison`
  - `results/lora/comparison_grids`
  - `data/synthetic/pending_review`
  - `data/synthetic/reviewed/accepted`
  - `data/synthetic/reviewed/rejected`
  - `data/synthetic/metadata`
- No se encontraron resultados, imagenes pendientes o imagenes aceptadas del Experimento B.

## 15. Estado del RAG y Streamlit

- RAG:
  - Hay implementacion basica en `src/rag/`: chunking, carga de `.pdf/.txt/.md`, embeddings con SentenceTransformers, FAISS y armado de prompt.
  - `src/pipelines/build_rag.py` recolecta chunks desde documentos.
  - `data/documents/` esta vacio salvo README y `.gitkeep` en subcarpetas.
  - `data/metadata/document_sources.csv` solo tiene encabezados.
  - `vector_db/` solo contiene README.
  - `results/rag/index/` solo contiene `.gitkeep`.
  - `scripts/build_vector_db.py` sigue como placeholder.
- Streamlit:
  - `app/streamlit_app.py` muestra titulo, uploader, campo de observaciones, imagen cargada y avisos.
  - El propio mensaje de la app indica que falta integrar el clasificador entrenado y el RAG.

## 16. Errores, TODO o NotImplementedError encontrados

- No se encontraron ocurrencias de `NotImplementedError`.
- Marcadores o placeholders relevantes encontrados:
  - `app/streamlit_app.py`: integra clasificador visual, RAG local e informe preliminar cuando los artefactos estan disponibles.
  - `scripts/prepare_lora_dataset.py`: indica que falta preparar el dataset LoRA despues del baseline visual.
  - `scripts/build_vector_db.py`: indica que falta construir la base vectorial despues de reunir documentos tecnicos.
  - `scripts/evaluate_system.py`: indica que se implementara despues de obtener predicciones visuales, resultados RAG y evaluaciones sinteticas.
  - `src/reports/report_generator.py`: usa "Pendiente de generacion basada en fuentes recuperadas."
  - `PROJECT_STATUS.md`, `README.md` y `data/metadata/dataset_sources.csv` contienen pendientes que no reflejan completamente los artefactos ya existentes.
  - `scripts/prepare_dataset.py` conserva al inicio un bloque antiguo desactivado (`if False`) con mensaje de pendiente, aunque debajo hay implementacion funcional.

## 17. Comandos exactos que deberian ejecutarse a continuacion

Comandos de verificacion inmediata del estado actual:

```powershell
git status --short --branch
python -m pytest -q
python scripts/check_environment.py
python scripts/validate_dataset_split.py --dataset-split data/metadata/dataset_split.csv
```

Comandos para reconciliar resultados ResNet18 existentes antes de reportar metricas finales:

```powershell
python scripts/evaluate_vision_model.py --config configs/vision_config.yaml --checkpoint models/vision/resnet18_baseline_best.pt
Get-Content results/vision/resnet18_baseline/metrics_test.json
Get-Content results/vision/resnet18_baseline/run_summary.json
```

Comandos de inspeccion antes de versionar cambios:

```powershell
git diff -- models/README.md
git status --short configs/synthetic_generation.yaml data/lora
```

Comandos que dependen de acciones pendientes, no ejecutables como siguiente paso automatico hasta completar insumos:

```powershell
python scripts/run_demo.py
```

No hay comando real listo para construir el vector DB desde `scripts/build_vector_db.py` porque el script inspeccionado es placeholder y el corpus documental esta vacio.
