# SeedCare-RAG LoRA: sistema multimodal para clasificar defectos visibles en semillas de soja y recuperar evidencia tecnica documental

**Autores:** Equipo del Proyecto Integrador ISIA-119, 2026-I. Completar nombres reales antes de la entrega final.

## Abstract

This work presents SeedCare-RAG LoRA, an academic prototype for visual classification of soybean seed defects, retrieval of technical documentary evidence, and documentation of a generative experiment using Stable Diffusion 1.5 with LoRA. The system receives an image of an individual soybean seed and estimates one of five visual categories: `intact`, `spotted`, `immature`, `broken`, or `skin_damaged`. The predicted category and optional observations are transformed into a retrieval query over a local RAG corpus built from registered documentary sources. The retrieved fragments are used to generate a preliminary deterministic report with confidence, source references, and explicit limitations. A ResNet18 baseline was trained and evaluated on real images. The canonical test result was accuracy 0.670498 and macro-F1 0.625955 over 522 test images. The RAG retrieval evaluation used 20 queries and obtained Hit@5 of 0.80 and MRR of 0.589167. Stable Diffusion 1.5 with LoRA was also trained, and local evidence confirms a 6.12 MB adapter and 1000 metadata records, but the evidence is partial because logs, hardware, training time, human visual evaluation, and base-vs-LoRA comparison are not available. The planned classifier comparison with synthetic images was postponed. The tool is not diagnostic and must not be used to infer fungi, pathogens, or disease from images.

**Index Terms:** soybean seeds, visual classification, ResNet18, retrieval augmented generation, FAISS, LoRA, Stable Diffusion, responsible AI.

## I. Introduccion

La revision visual de semillas de soja es una actividad relevante para control de calidad, seleccion de muestras y documentacion de hallazgos. En un flujo manual, la persona evaluadora observa atributos como roturas, manchas visibles, inmadurez, dano de cubierta y apariencia intacta. Esta tarea puede verse afectada por variabilidad entre evaluadores, condiciones de captura, volumen de imagenes y dificultad para vincular cada observacion con fuentes tecnicas verificables.

SeedCare-RAG LoRA surge como un prototipo academico para apoyar ese flujo. El sistema no intenta diagnosticar enfermedades ni identificar patogenos. Su objetivo es mas acotado: clasificar una imagen individual en una categoria visual definida por el dataset, recuperar evidencia documental relacionada y producir un informe preliminar con fuentes y limitaciones. La categoria `spotted`, en particular, se trata como una observacion visual. No se presenta como diagnostico de hongo.

El proyecto integra tres lineas tecnicas. La primera es vision computacional mediante transferencia de aprendizaje con ResNet18. La segunda es recuperacion documental mediante un indice FAISS construido sobre fragmentos de documentos tecnicos. La tercera es un experimento generativo con Stable Diffusion 1.5 ajustado mediante LoRA. Este ultimo se conserva como evidencia de entrenamiento generativo y como base para trabajo futuro, pero no se usa para afirmar una mejora del clasificador porque el Experimento B con imagenes sinteticas fue aplazado.

Las contribuciones del trabajo son:

1. Un pipeline reproducible para clasificar defectos visibles en cinco categorias de semillas de soja.
2. Una integracion RAG que recupera fragmentos documentales y evita presentar explicaciones sin fuentes.
3. Una consolidacion honesta de resultados reales, incluyendo discrepancias reconciliadas y pendientes.
4. Un marco de limitaciones eticas para no confundir categorias visuales con diagnosticos fitosanitarios.
5. Evidencia local de un entrenamiento LoRA, sin sobreinterpretar su impacto.

## II. Marco teorico

### A. Clasificacion visual y transferencia de aprendizaje

La transferencia de aprendizaje permite adaptar modelos de vision previamente entrenados a una tarea especifica con menos datos que un entrenamiento desde cero. En este proyecto se uso ResNet18 como baseline para clasificar cinco categorias visuales. La configuracion local registra cinco salidas, imagenes de 224 pixeles, dropout de 0.20, seleccion por macro-F1 y clases en el orden `intact`, `spotted`, `immature`, `broken`, `skin_damaged`.

El uso de macro-F1 es importante porque el interes no se limita al promedio global de aciertos. En tareas con varias categorias visuales, una accuracy aceptable puede ocultar bajo desempeno en una clase especifica. Por eso el reporte final incluye F1 por clase y no solo accuracy.

### B. RAG y recuperacion documental

Retrieval Augmented Generation, o RAG, combina una consulta con recuperacion de documentos para construir respuestas trazables. En este proyecto el componente de generacion se mantiene deterministico: el sistema organiza fragmentos recuperados y limitaciones sin invocar una evaluacion de calidad de LLM. El objetivo es que el informe preliminar muestre de donde proviene la informacion y evite recomendaciones no respaldadas.

El indice RAG usa embeddings `sentence-transformers/all-MiniLM-L6-v2`, fragmentos de tamano 700 con solapamiento 120, `top_k=5` y FAISS como almacen vectorial. La evaluacion mide recuperacion: Hit@k, MRR y precision@k en consultas evaluables.

### C. Stable Diffusion 1.5 y LoRA

LoRA permite adaptar partes entrenables de un modelo grande reduciendo el numero de parametros modificados. En este proyecto se registro un entrenamiento sobre Stable Diffusion 1.5 con rank 8, resolucion 512, batch 1, acumulacion de gradiente 4, learning rate 0.0001, precision mixta `fp16` y semilla 42. La evidencia local incluye un adaptador `.safetensors`, metadata y muestras copiadas.

El uso de LoRA se plantea como experimento de ampliacion sintetica. Sin embargo, las imagenes sinteticas no pueden incorporarse automaticamente. Deben ser revisadas por una persona y, si son aceptadas, solo pueden entrar al conjunto de entrenamiento. Nunca deben incorporarse a validacion ni prueba.

## III. Metodologia

### A. Dataset y preparacion

La fuente principal registrada es Soybean Seeds version 6, publicada en Mendeley Data con DOI `10.17632/v6vzvfszj6.6`. El repositorio conserva la regla de no modificar `data/raw/`. La preparacion copia datos validos hacia `data/processed/` y mantiene metadatos en `data/metadata/`.

La consolidacion final registra 5513 imagenes auditadas. De ellas, 5223 fueron incluidas y 290 excluidas por duplicado exacto. El split final contiene 4179 imagenes de entrenamiento, 522 de validacion y 522 de prueba. El resumen indica `synthetic_train_images=0`, por lo que el entrenamiento final documentado no incorpora sinteticos.

### B. Entrenamiento ResNet18

El baseline ResNet18 fue entrenado con imagenes reales. La configuracion principal esta en `configs/vision_config.yaml`. El checkpoint local esperado es `models/vision/resnet18_baseline_best.pt`, que no debe versionarse. La evaluacion canonica usa el split manifest `data/metadata/dataset_split.csv` y guarda resultados en `results/vision/resnet18_baseline/`.

Durante la revision final se encontro una discrepancia historica entre un valor alto de `test_macro_f1` en un resumen anterior y los archivos de evaluacion. El reporte de reconciliacion declara obsoleto el valor alto y confirma como resultado canonico el archivo `metrics_test.json`. Por tanto, el informe usa accuracy 0.670498 y macro-F1 0.625955 en test.

### C. Preparacion del RAG

El corpus documental esta registrado en `data/metadata/document_sources.csv`. El sistema no inventa autores, anos, licencias ni URLs cuando esos campos no estan disponibles. Los documentos aceptados cubren calidad general, dano mecanico, inmadurez, manchas visibles, dano de cubierta, almacenamiento y manejo.

El indice actual contiene 1316 chunks: 684 de `DOC001`, 12 de `DOC002`, 90 de `DOC003`, 441 de `DOC004`, 51 de `DOC005` y 38 de `DOC006`. La construccion usa los documentos aceptados y los metadatos locales para generar `vector_db/index.faiss` y `vector_db/metadata.json`.

### D. Integracion del sistema

La funcion `analyze_seed` orquesta el flujo de inferencia. Recibe una imagen, carga el checkpoint visual o usa recursos cacheados, produce una prediccion, calcula incertidumbre, construye una consulta, recupera fragmentos y genera un informe preliminar. La interfaz Streamlit en `app/app.py` permite cargar imagenes, ejecutar el analisis, revisar evidencia RAG, leer el informe y descargar un reporte.

El informe incluye limitaciones por defecto. Si la etiqueta es `spotted`, agrega una advertencia explicita de que la categoria describe alteraciones visibles y no confirma hongo ni enfermedad.

### E. Entrenamiento LoRA y aplazamiento del Experimento B

Stable Diffusion 1.5 + LoRA fue entrenado. La evidencia local esta consolidada en `results/lora/`. Se confirma un adaptador local de 6.12 MB y 1000 registros de metadata, distribuidos en 200 por cada categoria visual.

El Experimento B, que debia comparar ResNet18 entrenado con datos reales frente a ResNet18 entrenado con datos reales mas sinteticos aceptados, fue aplazado. La razon metodologica es que los sinteticos requieren revision humana previa y una comparacion controlada. Por ello no se reporta mejora por sinteticos.

## IV. Resultados

### A. Resultados del dataset

El pipeline de preparacion produjo un dataset depurado de 5223 imagenes. Se excluyeron 290 imagenes por duplicados exactos. No se incorporaron imagenes sinteticas al entrenamiento final documentado.

### B. Resultados del clasificador visual

La evaluacion test de ResNet18 se realizo sobre 522 imagenes. Los resultados canonicos son:

| metrica | valor |
| --- | ---: |
| accuracy | 0.670498 |
| macro precision | 0.741193 |
| macro recall | 0.650534 |
| macro-F1 | 0.625955 |

El desempeno por clase muestra diferencias relevantes:

| clase | soporte | F1 |
| --- | ---: | ---: |
| `intact` | 91 | 0.296296 |
| `spotted` | 106 | 0.724409 |
| `immature` | 112 | 0.688406 |
| `broken` | 100 | 0.641975 |
| `skin_damaged` | 113 | 0.778689 |

La clase `intact` es el principal punto debil del baseline. Esto limita cualquier uso operativo y justifica trabajo futuro en datos, augmentacion, revision de errores y arquitectura.

### C. Resultados RAG

La evaluacion de recuperacion uso 20 consultas. Los resultados fueron:

| metrica | valor |
| --- | ---: |
| Hit@1 | 0.450000 |
| Hit@3 | 0.700000 |
| Hit@5 | 0.800000 |
| MRR | 0.589167 |
| Precision@1 | 0.500000 |
| Precision@3 | 0.472222 |
| Precision@5 | 0.433333 |
| latencia media de recuperacion | 12.807 ms |

Cuatro consultas no recuperaron el documento esperado en top 5: `RAG009`, `RAG012`, `RAG017` y `RAG020`. La revision humana esta pendiente para las 20 consultas.

### D. Resultados del sistema integrado

La evaluacion final registro cinco casos de demo en el split de validacion. Los cinco casos terminaron exitosamente. El tiempo medio total fue 9.671234 segundos, con 8.850886 segundos de inferencia visual y 0.809960 segundos de recuperacion. El primer caso refleja carga en frio, por lo que el promedio no debe interpretarse como latencia estabilizada.

### E. Evidencia LoRA

La evidencia LoRA tiene estado `PARTIAL`. Se confirma el adaptador local, los parametros principales y la metadata de entrenamiento. No se confirma hardware, tiempo de entrenamiento, logs de notebook ni comparacion base vs. LoRA. Tampoco hay evaluacion humana de muestras sinteticas.

## V. Evaluacion critica

Los resultados muestran una integracion funcional, pero no un sistema listo para uso diagnostico. El componente visual tiene desempeno moderado y desigual. El RAG recupera documentos relevantes en la mayoria de consultas a top 5, pero falla en cuatro casos y aun no tiene revision humana. LoRA fue entrenado, pero la evidencia no basta para concluir mejora generativa ni impacto positivo en clasificacion.

La principal fortaleza es la trazabilidad. Cada resultado final apunta a archivos locales: metricas de vision, evaluacion RAG, evidencia LoRA y reporte consolidado. La principal debilidad es que varias dimensiones cualitativas permanecen pendientes: revision humana de RAG, validacion experta del informe y evaluacion visual de sinteticos.

La decision de aplazar el Experimento B evita una afirmacion no sustentada. Si se incorporaran sinteticos sin revision, podria introducirse ruido, reforzar sesgos visuales o contaminar la evaluacion. El proyecto documenta esa frontera de forma explicita.

## VI. Reflexion etica

El riesgo mas importante es comunicar una prediccion visual como diagnostico. Una fotografia de una semilla no confirma la presencia de un patogeno. En particular, una semilla marcada como `spotted` puede tener alteraciones visibles, pero eso no equivale a diagnosticar hongo o enfermedad. Por esta razon, la interfaz y el informe incluyen advertencias.

Otro riesgo es presentar recuperacion documental como recomendacion final. El RAG devuelve fragmentos del corpus, pero no garantiza cobertura completa de la literatura ni sustitucion de criterio profesional. La ausencia de una fuente en top 5 no implica que el tema no exista. La presencia de un fragmento tampoco implica que aplique automaticamente al caso observado.

El uso de datos sinteticos tambien exige cuidado. Las imagenes generadas pueden verse plausibles y aun asi ser incorrectas, ambiguas o sesgadas. Por eso el proyecto establece revision humana y uso exclusivo en `train` si son aceptadas. No se deben usar sinteticos para validation o test.

Finalmente, el proyecto evita completar bibliografia faltante con suposiciones. Los documentos del corpus se citan segun los campos registrados. Cuando faltan autores, ano o URL, esa ausencia se conserva.

## VII. Conclusiones

SeedCare-RAG LoRA demuestra una integracion academica de clasificacion visual, recuperacion documental e informe preliminar para semillas de soja. El ResNet18 fue entrenado y evaluado con resultados canonicos reconciliados. El RAG fue construido con fuentes documentales locales y evaluado en recuperacion. LoRA fue entrenado y cuenta con evidencia local parcial.

Los resultados apoyan la viabilidad del flujo, pero tambien muestran limites claros. El macro-F1 de 0.625955 y el F1 bajo de `intact` impiden presentar el clasificador como confiable para decisiones finales. El RAG necesita revision humana. La evidencia LoRA no permite afirmar mejora del clasificador. El Experimento B queda correctamente como trabajo futuro.

La contribucion principal es una base reproducible y eticamente delimitada: se clasifica lo visual, se recuperan fuentes y se advierte sobre limitaciones. Esa separacion es esencial para no transformar un prototipo academico en una herramienta diagnostica sin validacion.

## VIII. Trabajo futuro

El trabajo futuro debe priorizar:

1. Mejorar el clasificador, especialmente la clase `intact`.
2. Revisar manualmente errores de confusion y calidad de etiquetas.
3. Completar revision humana de las 20 consultas RAG.
4. Completar metadatos documentales desde fuentes verificadas.
5. Evaluar calidad visual y fidelidad de imagenes LoRA con criterios humanos.
6. Ejecutar Experimento B solo con sinteticos aceptados y manteniendo validation/test reales.
7. Medir robustez por iluminacion, fondo, camara y procedencia de imagen.
8. Incorporar validacion experta antes de cualquier uso fuera del entorno academico.

## Referencias

[1] W. Lin et al., "Soybean Seeds," Mendeley Data, Version 6, 2023. DOI: `10.17632/v6vzvfszj6.6`.

[2] PyTorch, "Transfer Learning for Computer Vision Tutorial," documentacion oficial.

[3] Hugging Face Diffusers, "LoRA training," documentacion oficial.

[4] OpenAI, "Codex CLI" y "Custom instructions with AGENTS.md," documentacion oficial.

[5] Documento de alcance del Proyecto Integrador ISIA-119, 2026-I.

[6] `DOC001`, "Genebank Standards for Plant Genetic Resources for Food and Agriculture," registrado en `data/metadata/document_sources.csv`.

[7] `DOC002`, "Subpart J -- United States Standards for Soybeans," registrado en `data/metadata/document_sources.csv`.

[8] `DOC003`, "Evaluation of soybean genotypes for reaction to natural field infection by Cercospora species causing purple seed stain," registrado en `data/metadata/document_sources.csv`.

[9] `DOC004`, "Field Book," registrado en `data/metadata/document_sources.csv`.

[10] `DOC005`, fuente documental registrada en `data/metadata/document_sources.csv`.

[11] `DOC006`, fuente documental registrada en `data/metadata/document_sources.csv`.
