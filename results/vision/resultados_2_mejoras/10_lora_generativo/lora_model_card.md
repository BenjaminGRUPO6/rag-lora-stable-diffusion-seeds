# Modelo generativo LoRA SD 1.5

Status: **PARTIAL**

El LoRA genera imágenes sintéticas de semillas. No clasifica la imagen cargada y no modifica la confianza del clasificador ResNet18.

## Que hace
- Genera imagenes sinteticas de semillas cuando se carga externamente en un pipeline SD 1.5.
- Registra evidencia de configuracion, metadata y adaptador local ya entrenado.

## Que no hace
- No clasifica imagenes cargadas en Streamlit.
- No modifica probabilidades ni confianza de ResNet18.
- No ejecuta generacion automatica ni masiva en esta etapa.

## Evidencia verificada
- Modelo base: `stable-diffusion-v1-5/stable-diffusion-v1-5`
- Trigger word: `soybeanseed`
- Resolucion: `512`
- Rank: `8`
- Learning rate: `0.0001`
- Pasos inicial/full: `100` / `800`
- Hardware: `EVIDENCE_MISSING`
- Adaptador: `soybean_sd15`
- Archivo adaptador: `pytorch_lora_weights.safetensors`
- Metadata: `1000` registros; `1000` imagenes existentes.

## Clases visuales
- intact: `200`
- broken: `200`
- spotted (categoria visual): `200`
- immature: `200`
- skin_damaged: `200`

## Evidencia faltante
- Base-vs-adapted comparison images are not available.
- Notebook outputs/logs are missing; hardware is not verified.
- Parameter not verified: hardware

## Aviso de uso
- Las imagenes sinteticas solo pueden incorporarse a `train` despues de revision humana.
- `spotted` se conserva como categoria visual, no como diagnostico de hongo.
