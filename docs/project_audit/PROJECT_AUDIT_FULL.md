# Auditoria tecnica completa del proyecto

Auditoria: 2026-07-14T02:12:47-05:00

## 1. Resumen ejecutivo
El proyecto clasifica defectos visibles en semillas de soja, recupera evidencia tecnica mediante RAG y documenta un adaptador LoRA SD 1.5. No se entreno, no se descargo, no se ejecuto Stable Diffusion y no se modificaron datasets/checkpoints/codigo funcional.

## 2. Estado general
|area|estado|evidencia|
|---|---|---|
|App|PASS|smoke test .venv|
|Vision|PASS|checkpoint + funcional|
|Dataset|VERIFICADO|dataset_split.csv|
|RAG|PASS|functional rag_status=faiss|
|LoRA|PARTIAL|hardware/comparacion NO VERIFICADA|
|Resultados 1|VERIFICADO|r1_metricas.json|
|Resultados 2|VERIFICADO|final_metrics.json|
|Pruebas|PASS_IN_VENV|pytest/smoke/functional|


## 3. Estructura
Ver `PROJECT_STRUCTURE_TREE.txt` y `PROJECT_STRUCTURE_EXPLAINED.md`. Tamano principal: 7332.67 MB.

## 4. Arquitectura
Flujo real documentado con evidencia en `app/streamlit_app.py`, `src/pipelines/analyze_seed.py`, `src/vision/inference_engine.py`, `src/rag/retrieval.py`, `src/reports/report_generator.py`.

## 5. Aplicacion
Entrypoint `app/streamlit_app.py`; runner `scripts/run_demo.py`; smoke PASS.

## 6. Clasificador
Produccion `resnet18_v2_tta_light`; checkpoint `models/vision/resnet18_v2_best.pt`; clases intact, spotted, immature, broken, skin_damaged.

## 7. Dataset
Total 5513; splits y clases en CSV; no se copiaron imagenes.

## 8. RAG
FAISS local con 1316 chunks y top-k 5.

## 9. LoRA
El LoRA es un adaptador generativo para Stable Diffusion. Su función es generar imágenes sintéticas de semillas. No clasifica la imagen cargada, no ejecuta el RAG y no aumenta directamente la confianza del clasificador.

## 10. Resultados 1
Accuracy 0.6704980842911877; macro-F1 0.6259550750897566.

## 11. Resultados 2
Accuracy 0.9176245210727969; macro-F1 0.9168669642726247; seleccionado por validation macro-F1.

## 12. Metricas
Soportes R1 522, R2 522; discrepancias 1.

## 13. Pruebas
`.venv` PASS: pytest, compileall, smoke y funcional. Global Python FAIL por dependencias.

## 14. Dependencias
91 paquetes inventariados; sin instalaciones.

## 15. GitHub
Rama feature/vision-v2-results-2; commit 2833d21; cambios locales preexistentes; no commit/push.

## 16. Tamano
Cinco carpetas mas pesadas: .venv, models, data, results, .git. Ver CSV/PNG.

## 17. Riesgos
|id|severity|area|finding|evidence|status|
|---|---|---|---|---|---|
|RISK-001|alto|entorno|Python global no tiene dependencias; usar .venv.|command_outputs/11_*|abierto|
|RISK-002|medio|git|Cambios locales preexistentes y muchos resultados no versionados.|git status inicial/final|abierto|
|RISK-003|medio|metricas|R1 incluye reconciliacion por metrica de validacion obsoleta archivada.|final_metrics.json limitations|documentado|
|RISK-004|medio|lora|Hardware y comparaciones base-vs-adaptado no verificadas.|lora_evidence.json|abierto|
|RISK-005|bajo|demo|Prueba funcional predijo skin_damaged para imagen broken; pipeline pasa pero existe error puntual.|functional_test.json|abierto|
|RISK-006|informativo|almacenamiento|Checkpoints y recuperaciones ocupan espacio.|largest_files.csv|documentado|


## 18. Limitaciones
No se verifico sincronizacion remota con fetch; hardware LoRA NO VERIFICADA; Python global falla.

## 19. Elementos terminados
App, vision, RAG, resultados R1/R2, pruebas y documentacion de ejecucion.

## 20. Elementos pendientes
Limpiar estado Git, completar evidencia LoRA, decidir versionado de resultados pesados.

## 21. Recomendaciones priorizadas
1. Usar `.venv`.
2. Separar cambios por rama/commit.
3. Mantener datasets/checkpoints/indices fuera de GitHub.
4. Revisar humanamente cualquier sintetico antes de train.

## 22. Comandos
Ver `HOW_TO_RUN_PROJECT.md`.

## 23. Indice de archivos generados
|archivo|
|---|
|docs/project_audit/DATASET_AUDIT.md|
|docs/project_audit/ENVIRONMENT_AUDIT.md|
|docs/project_audit/GIT_AND_GITHUB_AUDIT.md|
|docs/project_audit/HOW_TO_RUN_PROJECT.md|
|docs/project_audit/LORA_AUDIT.md|
|docs/project_audit/METRICS_AUDIT.md|
|docs/project_audit/PROJECT_STRUCTURE_EXPLAINED.md|
|docs/project_audit/PROJECT_STRUCTURE_TREE.txt|
|docs/project_audit/RAG_AUDIT.md|
|docs/project_audit/RISKS_AND_LIMITATIONS.md|
|docs/project_audit/STREAMLIT_APPLICATION.md|
|docs/project_audit/SYSTEM_ARCHITECTURE.md|
|docs/project_audit/TEST_AUDIT.md|
|docs/project_audit/VISION_MODEL_AUDIT.md|
|results/project_audit/classifier_rag_lora_comparison.png|
|results/project_audit/dataset_class_counts.csv|
|results/project_audit/dataset_class_distribution.png|
|results/project_audit/dataset_split_counts.csv|
|results/project_audit/dataset_split_distribution.png|
|results/project_audit/dataset_storage_map.png|
|results/project_audit/dataset_summary.json|
|results/project_audit/dependency_categories.png|
|results/project_audit/duplicate_files.csv|
|results/project_audit/environment_packages.csv|
|results/project_audit/environment_status.png|
|results/project_audit/environment_summary.json|
|results/project_audit/f1_by_class_current.png|
|results/project_audit/folder_sizes.csv|
|results/project_audit/functional_status.png|
|results/project_audit/generate_audit_artifacts.py|
|results/project_audit/git_content_categories.png|
|results/project_audit/git_ignored_summary.csv|
|results/project_audit/git_risk_summary.json|
|results/project_audit/git_tracked_large_files.csv|
|results/project_audit/git_tracked_summary.csv|
|results/project_audit/github_upload_readiness.png|
|results/project_audit/image_analysis_flow.png|
|results/project_audit/largest_files.csv|
|results/project_audit/largest_files.png|
|results/project_audit/lora_evidence.csv|
|results/project_audit/lora_inventory.json|
|results/project_audit/lora_role_in_project.png|
|results/project_audit/lora_training_summary.png|
|results/project_audit/metrics_consistency.json|
|results/project_audit/metrics_consistency_status.png|
|results/project_audit/metrics_file_comparison.png|
|results/project_audit/metrics_sources.csv|
|results/project_audit/module_dependency_map.png|
|results/project_audit/project_architecture.png|
|results/project_audit/project_completion_status.png|
|results/project_audit/project_inventory.csv|
|results/project_audit/project_inventory.json|
|results/project_audit/project_map.png|
|results/project_audit/project_size_by_folder.png|
|results/project_audit/project_status_dashboard.png|
|results/project_audit/rag_components.json|
|results/project_audit/rag_pipeline.png|
|results/project_audit/rag_sources.csv|
|results/project_audit/rag_sources_inventory.png|
|results/project_audit/results_1_inventory.png|
|results/project_audit/results_1_summary.json|
|results/project_audit/results_1_vs_results_2_status.png|
|results/project_audit/results_2_inventory.png|
|results/project_audit/results_2_summary.json|
|results/project_audit/results_inventory.csv|
|results/project_audit/risk_register.csv|
|results/project_audit/risk_severity_dashboard.png|
|results/project_audit/storage_composition.png|
|results/project_audit/system_components.csv|
|results/project_audit/test_commands.csv|
|results/project_audit/test_results.json|
|results/project_audit/test_status_dashboard.png|
|results/project_audit/vision_model_inventory.json|

