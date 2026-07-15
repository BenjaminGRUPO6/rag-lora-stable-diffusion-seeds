# Auditoria del clasificador ResNet18

Auditoria: 2026-07-14T02:12:47-05:00

- Arquitectura: `resnet18`.
- Clases: `intact, spotted, immature, broken, skin_damaged`.
- Entrada: RGB `224` px, normalizacion ImageNet.
- Checkpoint: `models/vision/resnet18_v2_best.pt`.
- SHA-256: `5368128d6bbe47a7e11e5b440116211c0e2a0a8a91c19a285944ddd641112e5b`.
- Parametros desde state_dict: `NO_VERIFICADA`.
- Salida: label, confidence, probabilities, logits, top_3, segunda clase, margen y calibracion.
- Dispositivo: CPU/CUDA si disponible.
- `spotted` es categoria visual, no diagnostico de hongo.
