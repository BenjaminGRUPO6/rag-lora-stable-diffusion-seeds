# Arquitectura final

## Objetivo

SeedCare-RAG LoRA integra vision computacional, recuperacion documental y evidencia generativa para apoyar la revision visual de semillas de soja. El sistema estima una categoria visual, recupera fuentes tecnicas relacionadas y produce un informe preliminar con advertencias.

La herramienta no es diagnostica. La categoria `spotted` describe una alteracion visible y no confirma hongo, enfermedad ni patogeno.

## Flujo de inferencia

```text
Imagen JPG/PNG
  -> validacion de imagen
  -> transformacion a 224x224
  -> ResNet18 ajustado
  -> etiqueta, confianza y probabilidades
  -> regla de incertidumbre por confianza y margen
  -> consulta documental desde etiqueta y observaciones
  -> embeddings all-MiniLM-L6-v2
  -> recuperacion FAISS top_k=5
  -> informe preliminar deterministico
```

## Componentes

| componente | implementacion | evidencia |
| --- | --- | --- |
| Vision | ResNet18 preentrenado ajustado a cinco clases | `src/vision/`, `configs/vision_config.yaml`, `results/vision/resnet18_baseline/` |
| RAG | Chunking, embeddings y FAISS | `src/rag/`, `configs/rag.yaml`, `vector_db/` |
| Pipeline | Orquestacion de vision, recuperacion e informe | `src/pipelines/analyze_seed.py` |
| Informe | Reporte deterministico con fuentes y limitaciones | `src/reports/report_generator.py` |
| Demo | Interfaz Streamlit | `app/app.py`, `scripts/run_demo.py` |
| LoRA | Adaptador Stable Diffusion 1.5 entrenado, evidencia parcial | `configs/lora_sd15.yaml`, `results/lora/` |

## Datos y fuentes

El dataset principal es Soybean Seeds version 6. Los datos originales se conservan fuera de Git en `data/raw/` y no se modifican. El split depurado se registra en `data/metadata/dataset_split.csv`.

El RAG utiliza documentos aceptados registrados en `data/metadata/document_sources.csv`. El indice actual contiene 1316 chunks en seis documentos (`DOC001` a `DOC006`). La metadata bibliografica incompleta se conserva incompleta para no inventar autores, fechas o licencias.

## Justificacion tecnica

ResNet18 se eligio como baseline por ser una arquitectura estable para transferencia de aprendizaje y viable en equipos de recursos moderados. RAG se usa para separar la prediccion visual de la explicacion tecnica: el clasificador estima una categoria y el recuperador aporta evidencia documental. LoRA se usa como experimento generativo de bajo costo relativo para explorar ampliacion sintetica sin reentrenar completamente Stable Diffusion 1.5.

## Controles de seguridad

- Umbrales de incertidumbre: confianza minima 0.60 y margen minimo 0.15.
- Informe con limitaciones obligatorias.
- Abstencion o advertencia cuando no hay evidencia recuperada.
- Sinteticos excluidos de validation y test.
- Incorporacion de sinteticos a `train` solo despues de revision humana.
- Pesos, datasets, indices, caches y secretos fuera de Git.
