# SeedCare-RAG LoRA

**Repositorio:** `rag-lora-stable-diffusion-seeds`

## Título final

**SeedCare-RAG LoRA: Sistema multimodal para identificar posibles daños físicos, biológicos y morfológicos en semillas, recuperar métodos de prevención y manejo, y ampliar clases minoritarias mediante Stable Diffusion 1.5 ajustado con LoRA**

## Descripción puntual

El sistema recibe una fotografía de una semilla y utiliza un modelo visual ajustado con el dataset del proyecto para estimar si la muestra es sana o presenta un posible daño físico, biológico o morfológico. A partir del resultado, un módulo RAG recupera información desde artículos, manuales, fichas técnicas y guías especializadas sobre causas, síntomas, prevención y métodos de manejo. Finalmente, se genera un informe técnico con fuentes, nivel de confianza y limitaciones. Como parte central del entrenamiento generativo, Stable Diffusion 1.5 se adapta mediante LoRA para generar ejemplos sintéticos de clases minoritarias; estos ejemplos se revisan antes de incorporarse únicamente al conjunto de entrenamiento.

> El sistema no reemplaza un diagnóstico de laboratorio ni la evaluación de un especialista. Sus resultados se presentan como apoyo preliminar y deben estar acompañados de fuentes y supervisión humana.

## ¿Dónde se cumple el entrenamiento de modelos?

El proyecto incluye dos procesos de entrenamiento medibles:

1. **Ajuste del modelo visual:** fine-tuning de ResNet18 o EfficientNet-B0 para clasificar `healthy`, `physical_damage`, `biological_damage` y `morphological_damage`.
2. **Entrenamiento LoRA de Stable Diffusion 1.5:** adaptación del UNet mediante capas LoRA usando imágenes y captions del dominio de semillas. El adaptador se usa para generar datos sintéticos de categorías con pocas muestras.

El RAG no sustituye estos entrenamientos: su función es recuperar evidencia documental y reducir respuestas sin respaldo.

## Arquitectura

```text
Imagen de semilla
      |
      v
Modelo visual ajustado
      |
      +--> clase estimada + confianza
      |
      v
Consulta automática al RAG
      |
      v
Recuperación de fuentes técnicas
      |
      v
Informe: observaciones, posibles causas, prevención, manejo y fuentes

Módulo de datos sintéticos (entrenamiento):
clases minoritarias -> SD 1.5 + LoRA -> imágenes sintéticas -> revisión humana -> train
```

## Estructura del repositorio

```text
app/                 Interfaz Streamlit
configs/             Parámetros del proyecto
src/data/            Auditoría, limpieza, división y aumento
src/vision/          Entrenamiento e inferencia del clasificador
src/rag/             Carga, embeddings, FAISS y recuperación
src/reports/         Construcción del informe técnico
src/synthetic_data/  Preparación, entrenamiento LoRA y filtrado sintético
src/pipelines/       Integración de los módulos
notebooks/           Experimentos ordenados
scripts/             Entradas ejecutables
codex/               Prompts de trabajo para Codex local
 docs/                Plan, arquitectura, GitHub y metodología
```

## Inicio rápido local

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-core.txt
python scripts/check_environment.py
python -m pytest -q
```

Para la interfaz:

```powershell
pip install -r requirements-app.txt
streamlit run app/app.py
```

El entrenamiento LoRA se realiza preferentemente en Google Colab con GPU. Consulte `notebooks/06_entrenamiento_lora_sd15_colab.ipynb` y `docs/02_PLAN_PASO_A_PASO.md`.

## Flujo de trabajo

1. Auditar y clasificar el dataset.
2. Definir etiquetas verificadas.
3. Separar `train/validation/test` antes de aumentar.
4. Entrenar y evaluar el modelo visual.
5. Preparar captions y entrenar LoRA de SD 1.5.
6. Revisar y filtrar imágenes sintéticas.
7. Construir el RAG documental.
8. Integrar el informe y la interfaz.
9. Comparar línea base y modelos ajustados.

## Documentos clave

- `docs/INFORME_PLAN_PROYECTO.docx`
- `docs/00_RESUMEN_EJECUTIVO.md`
- `docs/02_PLAN_PASO_A_PASO.md`
- `docs/03_CUMPLIMIENTO_ENTRENAMIENTO.md`
- `docs/04_GITHUB_DESDE_CERO.md`
- `codex/00_PROMPT_MAESTRO.md`

## Equipo

- Sebastián Preciado Peralta
- Carlos Mezones Burgos
- Luis Reyes Guerrero

## Seguridad de archivos

No subir a GitHub:

- dataset completo;
- `.env` o tokens;
- checkpoints y pesos grandes;
- índices vectoriales regenerables;
- cachés de Hugging Face, PyTorch o Jupyter.
