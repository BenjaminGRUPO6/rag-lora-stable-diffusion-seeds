# Como ejecutar, probar y reproducir el proyecto

1. `cd D:\Users\Luis\Documents\GitHub\rag-lora-stable-diffusion-seeds`
2. `.\.venv\Scripts\Activate.ps1`
3. `python --version`
4. `python -m pytest -q`
5. `python scripts/smoke_test_app.py --timeout 30 --output-dir results/project_audit/manual_smoke_test`
6. `python scripts/run_functional_test.py --output-dir results/project_audit/manual_functional_test`
7. `python scripts/run_demo.py --port 8501`
8. Abrir `http://127.0.0.1:8501`.
9. Detener con `Ctrl+C`.
10. Si el puerto esta ocupado: `python scripts/run_demo.py --port 8502`.
11. Checkpoint: `configs/production_vision_model.yaml` y `models/vision/resnet18_v2_best.pt`.
12. RAG: `vector_db/build_manifest.json`, `index.faiss`, `metadata.json`.
13. Resultados: `results/vision/resultados_1_baseline/` y `results/vision/resultados_2_mejoras/final/`.
14. No usar entrenamiento, descarga de modelos, Stable Diffusion, generacion LoRA, `git reset`, `git clean` ni modificar datos/checkpoints.
