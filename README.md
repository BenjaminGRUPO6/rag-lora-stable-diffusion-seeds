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
