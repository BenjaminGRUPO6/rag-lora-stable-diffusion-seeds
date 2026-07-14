# Auditoria LoRA y Stable Diffusion

Auditoria: 2026-07-14T02:12:47-05:00

El LoRA es un adaptador generativo para Stable Diffusion. Su función es generar imágenes sintéticas de semillas. No clasifica la imagen cargada, no ejecuta el RAG y no aumenta directamente la confianza del clasificador.

- Base: `stable-diffusion-v1-5/stable-diffusion-v1-5`.
- Trigger: `soybeanseed`.
- Resolucion `512`, rank `8`, LR `0.0001`, pasos `800`.
- Hardware: NO VERIFICADA.
- Adaptador: `models/lora/soybean_sd15/pytorch_lora_weights.safetensors`.
- Streamlit muestra evidencia y no carga Stable Diffusion.
- No hay evidencia de uso para reentrenar el clasificador; `final_metrics.json` reporta no sinteticos.
- Sinteticos a train solo despues de revision humana.
