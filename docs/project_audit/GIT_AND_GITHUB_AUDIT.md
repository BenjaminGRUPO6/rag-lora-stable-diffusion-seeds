# Auditoria Git y GitHub

Auditoria: 2026-07-14T02:12:47-05:00

- Rama: `feature/vision-v2-results-2`.
- Commit: `2833d21dac2c4819669e5206fe434adf338afd45`.
- Remotos:
```
origin	https://github.com/BenjaminGRUPO6/rag-lora-stable-diffusion-seeds.git (fetch)
origin	https://github.com/BenjaminGRUPO6/rag-lora-stable-diffusion-seeds.git (push)
```
- Ultimos commits:
```
2833d21 (HEAD -> feature/vision-v2-results-2) Merge branch 'main' of https://github.com/BenjaminGRUPO6/rag-lora-stable-diffusion-seeds into feature/vision-v2-results-2
fc086b6 (origin/feature/vision-v2-results-2) version mejorada
3b0c1c0 correcciones
6169f9c versiÃ³n nueva
a6c3647 mejoras....
385ca79 correcciones ....
191b2de (origin/main, origin/HEAD, main) Merge pull request #8 from BenjaminGRUPO6/feature/vision-v2-results-2
269a77d versiÃ³n 2
e3a2a78 fix: reconcile canonical results 1 metrics
fb43575  prepare results 1 and results 2 experiment structure
```
- Ramas:
```
backup/before-final-integration
  backup/estructura-anterior
  backup/functional-before-vision-v2
  feature/dataset-audit
  feature/dataset-cleaning-split
  feature/dataset-soybean-seeds
  feature/experiment-b-intact-broken
  feature/final-rag-integration
  feature/resnet18-baseline-training
  feature/sd15-lora-training
* feature/vision-v2-results-2
  fix/end-to-end-functional-audit
  fix/lora-colab-training
  main
  refactor/seedcare-rag-lora-final
  remotes/origin/HEAD -> origin/main
  remotes/origin/backup/estructura-anterior
  remotes/origin/feature/dataset-audit
  remotes/origin/feature/dataset-cleaning-split
  remotes/origin/feature/dataset-soybean-seeds
  remotes/origin/feature/experiment-b-intact-broken
  remotes/origin/feature/final-rag-integration
  remotes/origin/feature/resnet18-baseline-training
  remotes/origin/feature/sd15-lora-training
  remotes/origin/feature/vision-v2-results-2
  remotes/origin/fix/end-to-end-functional-audit
  remotes/origin/fix/lora-colab-training
  remotes/origin/main
  remotes/origin/refactor/seedcare-rag-lora-final
```
- Objetos:
```
count: 938
size: 29.35 MiB
in-pack: 0
packs: 0
size-pack: 0 bytes
prune-packable: 0
garbage: 0
size-garbage: 0 bytes
```

Estado inicial observado antes de crear artefactos: rama feature/vision-v2-results-2 con cambios locales preexistentes en .gitignore, README/PROJECT_STATUS, app/streamlit_app.py, docs, resultados, src/vision, src/pipelines y tests; multiples archivos no versionados de Resultados 2. La auditoria no modifico esos archivos.

No se leyo `.env`, no se hizo commit/push/merge/reset/clean.
