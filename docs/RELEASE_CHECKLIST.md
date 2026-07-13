# Release checklist

Fecha local: 2026-07-13.
Estado sugerido: **WARNING**, listo para revision final sin blockers tecnicos.

## Codigo

- [x] Ejecutar `python -m pytest -q`.
- [x] Confirmar 52 pruebas aprobadas.
- [x] Revisar imports core con `python -B`.
- [x] Mantener compatibilidad Windows y Python 3.10/3.11.
- [x] No modificar `data/raw/`.
- [ ] Resolver o documentar el placeholder `scripts/prepare_lora_dataset.py` antes de presentarlo como modulo operativo.

## Datos

- [x] Validar split con `python scripts/validate_dataset_split.py --dataset-split data/metadata/dataset_split.csv`.
- [x] Confirmar `synthetic_train_images=0`.
- [x] Confirmar que `validation` y `test` no dependen de sinteticos.
- [x] Verificar que datasets e imagenes no esten rastreados por Git.
- [x] Mantener `data/raw/` inmutable.

## Modelos

- [x] Confirmar checkpoint visual local `models/vision/resnet18_baseline_best.pt`.
- [x] Confirmar que checkpoints `.pt/.pth/.ckpt` no esten versionados.
- [x] Confirmar que pesos `.safetensors` no esten versionados.
- [x] Documentar que los pesos son artefactos locales regenerables/no subibles.

## RAG

- [x] Confirmar fuentes locales en `data/documents/accepted/`.
- [x] Confirmar indice local `vector_db/index.faiss`.
- [x] Confirmar metadata local `vector_db/metadata.json`.
- [x] Confirmar metricas de recuperacion en `results/rag/evaluation/metrics.json`.
- [ ] Completar revision humana de las 20 consultas RAG.
- [ ] Revisar fallos `RAG009`, `RAG012`, `RAG017`, `RAG020`.
- [ ] Completar campos bibliograficos vacios solo desde fuentes verificadas.

## App

- [x] Ejecutar smoke test `python scripts/run_demo.py --timeout 25`.
- [x] Confirmar que Streamlit inicia y responde en healthcheck.
- [x] Mantener advertencia de no diagnostico visible.
- [x] Usar `python scripts/run_demo.py --serve` para demo interactiva.

## Metricas

- [x] Confirmar coherencia entre README y `results/system/final_metrics.json`.
- [x] Confirmar ResNet18 test: accuracy `0.670498`, macro-F1 `0.625955`.
- [x] Confirmar RAG: Hit@5 `0.800000`, MRR `0.589167`.
- [x] Confirmar sistema integrado: 5/5 casos de demo exitosos.
- [ ] Explicar en sustentacion el F1 bajo de `intact` (`0.296296`).

## LoRA

- [x] Confirmar adaptador local `models/lora/soybean_sd15/pytorch_lora_weights.safetensors`.
- [x] Confirmar metadata LoRA: 1000 registros, 200 por categoria visual.
- [x] Confirmar evidencia local parcial en `results/lora/evidence_report.md`.
- [x] Confirmar que no se ejecuto entrenamiento durante auditoria.
- [ ] Completar evidencia faltante: logs, hardware, tiempo de entrenamiento, comparacion base vs LoRA y evaluacion humana.
- [ ] Mantener sinteticos fuera de `train` hasta revision humana.

## Etica

- [x] Declarar que el sistema no diagnostica enfermedades ni hongos.
- [x] Declarar que `spotted` es solo categoria visual.
- [x] Declarar que el sistema no reemplaza laboratorio ni evaluacion especializada.
- [x] Declarar revision humana obligatoria para imagenes sinteticas.
- [x] Evitar recomendaciones fitosanitarias concluyentes.

## Informe

- [x] README actualizado con resultados canonicos.
- [x] `docs/ETHICS_AND_LIMITATIONS.md` existe.
- [x] `docs/REPRODUCIBILITY.md` existe.
- [x] `docs/11_REFERENCIAS_BASE.md` existe.
- [ ] Corregir mojibake y ampliar referencias verificadas si el informe final lo requiere.

## Demo

- [x] Comando interactivo definido:

```powershell
python scripts/run_demo.py --serve
```

- [x] Comando de contingencia CLI documentado en README.
- [x] Casos de demo consolidados en `results/system/demo_cases.csv`.
- [ ] Antes de presentar, confirmar que los artefactos locales ignorados existen en la maquina usada para la demo.

## GitHub

- [x] Trabajar fuera de `main`: rama actual `feature/final-rag-integration`.
- [x] Confirmar que datasets, pesos, indices, caches, tokens y `.env` no estan versionados.
- [x] Mantener `.gitignore` con reglas para artefactos pesados y secretos.
- [ ] Revisar `git status --short` antes de commit.
- [ ] Commit sugerido: `docs: add final audit and release checklist`.
- [ ] PR debe incluir comandos de prueba y resultados.
