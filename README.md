# SeedCare-RAG LoRA

Sistema multimodal para clasificar defectos visibles en semillas de soja, recuperar evidencia tecnica mediante RAG y documentar un experimento generativo con Stable Diffusion 1.5 ajustado con LoRA.

La herramienta es de apoyo visual y documental. No es diagnostica, no identifica patogenos y no reemplaza una evaluacion especializada o de laboratorio. La categoria `spotted` se usa solo como categoria visual.

## Resultados 2 - comparacion EfficientNet-B0 vs ResNet18 V2

Se corrigio un cuello de botella del entrenamiento EfficientNet-B0: el recorte automatico y la calidad visual se ejecutaban dentro del `__getitem__`, repitiendo segmentacion, morfologia, componentes conectados y blur por imagen y por epoca. La correccion precalcula crops regenerables en `data/cache/vision_crops/`, desactiva calidad durante entrenamiento y mantiene fallback al crop cuadrado de la imagen original si falta un crop.

Benchmark DataLoader en NVIDIA GeForce GTX 1050, 128 imagenes, batch 8:

- recorte en tiempo real, `num_workers=0`: 3.85 imagenes/s;
- cache, `num_workers=0`: 105.60 imagenes/s;
- cache, `num_workers=2`: 5.35 imagenes/s en Windows.

La configuracion elegida fue batch 8, `num_workers=0`, `pin_memory=true`, transferencias `non_blocking` en CUDA y cache activa. EfficientNet-B0 entreno 25 epocas con seed 42 y obtuvo validation macro-F1 0.899532 y test macro-F1 0.868306. La comparacion controlada mantuvo ResNet18 V2 frente a EfficientNet-B0 por validation macro-F1. La configuracion final de produccion agrega TTA ligera seleccionada por validation, con validation macro-F1 0.926503 y test macro-F1 final 0.916867. Artefactos: `results/vision/resultados_2_mejoras/final/`.

## Problema empresarial

El control de calidad de semillas requiere revisar atributos visibles como roturas, inmadurez, manchas, dano de cubierta y semillas aparentemente intactas. En escenarios operativos, esa revision puede ser lenta, variable entre evaluadores y dificil de documentar con evidencia tecnica. El proyecto propone un flujo reproducible que:

- estima una categoria visual desde una fotografia individual de semilla de soja;
- recupera fragmentos documentales relacionados con calidad, almacenamiento, manejo y posibles factores descritos en fuentes tecnicas;
- genera un informe preliminar con confianza, fuentes y limitaciones;
- conserva evidencia de entrenamiento de un adaptador LoRA para exploracion de datos sinteticos.

## Estado actual

- Repositorio y estructura: preparados.
- Dataset: **completado**.
- Baseline ResNet18: **entrenado y reconciliado**.
- Resultados 2: **consolidado**; produccion seleccionada como `resnet18_v2_tta_light`.
- Stable Diffusion 1.5 + LoRA: **entrenado**; evidencia parcial consolidada.
- Experimento B con imagenes sinteticas en ResNet18: **aplazado** como trabajo futuro.
- Corpus documental e indice del RAG: **disponibles localmente**.
- Aplicacion Streamlit: **integrada y auditada funcionalmente**.

## Dataset principal previsto

- Nombre: Soybean Seeds, version 6.
- Fuente: Mendeley Data.
- DOI: `10.17632/v6vzvfszj6.6`.
- Total publicado: 5513 imagenes individuales.
- Clases: `intact`, `spotted`, `immature`, `broken`, `skin_damaged`.
- Licencia: CC BY 4.0.

La etiqueta `spotted` describe una anomalia visible; no confirma por si sola hongos o una enfermedad especifica.

## Entrenamientos y evaluacion

1. **Fine-tuning visual:** baseline ResNet18 y Resultados 2 para clasificar las cinco categorias.
2. **Stable Diffusion 1.5 + LoRA:** ajuste generativo documentado como evidencia parcial.
3. **Trabajo futuro:** segundo entrenamiento con datos sinteticos aceptados despues de revision humana.

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
- Evidencia consolidada: `results/vision/resultados_2_mejoras/10_lora_generativo/`

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

Los checkpoints no deben subirse a Git. Las metricas, CSV y graficos seleccionados del
entrenamiento real pueden versionarse si son necesarios para documentar el experimento.

## Arquitectura

```text
Imagen de semilla
  -> validacion y preprocesamiento
  -> ResNet18 ajustado
  -> categoria visual, confianza e incertidumbre
  -> consulta RAG construida desde la categoria y observaciones
  -> embeddings + FAISS
  -> fragmentos documentales top-k
  -> informe preliminar deterministico con fuentes y limitaciones
```

Componentes principales:

- `src/vision/`: dataset, modelo ResNet18, entrenamiento, evaluacion e inferencia.
- `src/rag/`: carga documental, chunking, embeddings, FAISS y recuperacion.
- `src/pipelines/`: integracion de clasificador, RAG e informe.
- `app/`: demo Streamlit.
- `src/synthetic_data/`: preparacion y utilidades para Stable Diffusion 1.5 + LoRA.
- `results/`: metricas y reportes finales versionables.

## Instalacion

Requisitos: Windows, Python 3.10 o 3.11 y PowerShell.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-core.txt
pip install -r requirements-vision.txt
pip install -r requirements-rag.txt
pip install -r requirements-app.txt
```

Verificacion base:

```powershell
python scripts/check_environment.py
python -m pytest -q
```

La demo oficial queda activa hasta presionar `Ctrl+C`:

```powershell
python scripts/run_demo.py --port 8501
```

Tambien puede iniciarse directamente con Streamlit, sin configurar `PYTHONPATH`:

```powershell
python -m streamlit run app/streamlit_app.py --server.port 8501
```

El unico entrypoint Streamlit mantenido es `app/streamlit_app.py`. El entrypoint actual inserta la
raiz del repositorio en `sys.path` antes de importar `app.*` o `src.*`, una solucion estable en
Windows tanto desde la raiz como desde otro directorio.

## Configuracion

Archivos principales:

- `configs/vision_config.yaml`: arquitectura `resnet18`, cinco clases, umbrales de inferencia y rutas de resultados.
- `configs/rag.yaml`: `chunk_size=700`, `chunk_overlap=120`, `top_k=5`, embeddings `sentence-transformers/all-MiniLM-L6-v2` e indice FAISS.
- `configs/lora_sd15.yaml`: Stable Diffusion 1.5, LoRA rank 8, resolucion 512, `learning_rate=0.0001`, batch 1, acumulacion 4, semilla 42.

Artefactos locales esperados para ejecutar el sistema completo:

- checkpoint visual final: `models/vision/resnet18_v2_best.pt`;
- configuracion de produccion: `configs/production_vision_model.yaml`;
- indice RAG: `vector_db/index.faiss` y `vector_db/metadata.json`;
- documentos aceptados: `data/documents/accepted/`;
- metadatos documentales: `data/metadata/document_sources.csv`.

Los pesos, datasets completos, indices vectoriales, caches, tokens y `.env` no deben subirse a Git.

## Preparacion del corpus

El RAG utiliza fuentes documentales registradas en `data/metadata/document_sources.csv`. El corpus actual contiene 6 documentos aceptados (`DOC001` a `DOC006`) sobre estandares de germoplasma, estandares de soja, manchas visibles, campo, maduracion/desecacion e imbibicion. Algunos campos bibliograficos permanecen vacios porque no estan registrados localmente y no se inventan.

Para preparar o validar un corpus desde documentos locales:

```powershell
python scripts/prepare_rag_corpus.py `
  --input data/documents/inbox `
  --accepted data/documents/accepted `
  --rejected data/documents/rejected `
  --metadata data/metadata/document_sources.csv `
  --results results/rag
```

## Construccion del indice

```powershell
python scripts/build_vector_db.py `
  --config configs/rag.yaml `
  --documents data/documents/accepted `
  --sources data/metadata/document_sources.csv `
  --output vector_db
```

El indice actual contiene 1316 chunks distribuidos asi: `DOC001=684`, `DOC002=12`, `DOC003=90`, `DOC004=441`, `DOC005=51`, `DOC006=38`.

## Ejecucion de la demo

Demo Streamlit:

```powershell
python scripts/run_demo.py --serve
```

El comando imprime una URL local como `http://localhost:<puerto>`.

Ejecucion CLI para contingencia:

```powershell
python scripts/analyze_seed.py `
  --image data/processed/validation/immature/1.jpg `
  --output results/reports/demo_cli_report.json `
  --device cpu
```

## Resultados reales

Fuente consolidada de vision: `results/vision/resultados_2_mejoras/final/final_metrics.json`.

### Dataset

- Dataset publicado: Soybean Seeds version 6, Mendeley Data, DOI `10.17632/v6vzvfszj6.6`.
- Archivos auditados: 5513.
- Imagenes incluidas tras depuracion: 5223.
- Imagenes excluidas: 290 por duplicado exacto.
- Split: train 4179, validation 522, test 522.
- Imagenes sinteticas en train: 0.

### Vision: Resultados 1 vs Resultados 2 final

La seleccion final se hace por validation macro-F1. El split `test` se reporta solo como evaluacion final.

| configuracion | validation macro-F1 | test macro-F1 | test accuracy | decision |
| --- | ---: | ---: | ---: | --- |
| Resultados 1 - ResNet18 baseline | 0.660513 | 0.625955 | 0.670498 | no seleccionado |
| Resultados 2 - ResNet18 V2 | 0.920701 | 0.903329 | 0.904215 | no seleccionado |
| Resultados 2 - EfficientNet-B0 | 0.899532 | 0.868306 | 0.869732 | no seleccionado |
| Resultados 2 final - ResNet18 V2 + TTA light | 0.926503 | 0.916867 | 0.917625 | seleccionado |

Mejora de Resultados 2 final frente a Resultados 1 en test macro-F1: 0.290912 absoluta, 46.47% relativa. No se afirma mejora de latencia frente a R1 porque R1 no registro latencia comparable.

El reporte `results/vision/resultados_1_baseline/r1_reconciliation_report.md` declara como obsoleto un valor anterior alto de `test_macro_f1`; por eso Resultados 1 usa `macro-F1=0.625955`.

### RAG

Evaluacion de recuperacion solamente; no evalua generacion por LLM.

| metrica | valor |
| --- | ---: |
| consultas | 20 |
| Hit@1 | 0.450000 |
| Hit@3 | 0.700000 |
| Hit@5 | 0.800000 |
| MRR | 0.589167 |
| Precision@1 | 0.500000 |
| Precision@3 | 0.472222 |
| Precision@5 | 0.433333 |
| latencia media de recuperacion | 12.807 ms |

Consultas fallidas en Hit@5: `RAG009`, `RAG012`, `RAG017`, `RAG020`. La revision humana del RAG esta pendiente para las 20 consultas.

### LoRA

Stable Diffusion 1.5 + LoRA fue entrenado y existe evidencia local parcial:

- adaptador local: `models/lora/soybean_sd15/pytorch_lora_weights.safetensors`;
- tamano del adaptador: 6.12 MB;
- metadata LoRA: 1000 registros, 200 por cada categoria visual;
- estado de evidencia: `PARTIAL`;
- no se ejecuto reentrenamiento durante la consolidacion final;
- faltan logs del notebook, hardware, tiempo de entrenamiento, comparacion base vs. LoRA y evaluacion visual humana.

### Sistema integrado

Cinco casos de demo en validation se ejecutaron exitosamente. Promedios registrados:

- inferencia visual: 8.850886 s;
- recuperacion: 0.809960 s;
- total: 9.671234 s.

El primer caso incluye costo de carga en frio y eleva el promedio.

## Limitaciones

- El sistema clasifica categorias visuales, no diagnostica enfermedades ni hongos.
- `spotted` no significa diagnostico fungico; solo indica una alteracion visible.
- La precision del clasificador no es perfecta; persisten errores por clase y casos de alta confianza incorrecta.
- El RAG depende de seis fuentes documentales locales y su revision humana esta pendiente.
- Algunas fuentes del corpus tienen metadatos bibliograficos incompletos; no se completan con suposiciones.
- La evaluacion RAG mide recuperacion, no calidad de respuesta de un LLM.
- La evidencia LoRA es parcial y no incluye comparacion base vs. LoRA.
- El Experimento B con imagenes sinteticas fue aplazado.
- Las imagenes sinteticas solo pueden incorporarse a `train` despues de revision humana; no deben usarse en `validation` ni `test`.

## Estructura del repositorio

```text
app/                 Demo Streamlit.
configs/             Configuraciones de dataset, vision, RAG, LoRA y reportes.
data/                Datos locales y metadatos; data/raw es inmutable y no versionable.
docs/                Documentacion tecnica, sustentacion e informe.
models/              Checkpoints y pesos locales no versionables.
notebooks/           Notebooks por etapa.
results/             Metricas, reportes y salidas versionables seleccionadas.
scripts/             Entrypoints CLI reproducibles.
src/                 Codigo fuente del sistema.
tests/               Pruebas automatizadas.
vector_db/           Indice FAISS local regenerable, no versionable.
```

## Referencias registradas

Las referencias bibliograficas permitidas estan registradas en `docs/11_REFERENCIAS_BASE.md` y las fuentes del RAG en `data/metadata/document_sources.csv`. Este README no agrega bibliografia externa.
