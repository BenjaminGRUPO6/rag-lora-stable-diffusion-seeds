# Aplicacion Streamlit

Auditoria: 2026-07-14T02:12:47-05:00

- Entrypoint oficial: `app/streamlit_app.py`.
- Comando oficial: `python scripts/run_demo.py --port 8501`.
- Puerto por defecto: `8501`.
- `app/app.py`: no existe.
- Tabs: A Analisis, B Explicabilidad, C Evidencia RAG, D Resultados 1 vs Resultados 2, Modelo generativo LoRA.
- Carga JPG/JPEG/PNG, preprocesa con `preprocess_image`, cachea modelos con `st.cache_resource`, maneja errores con `render_error` y exporta PNG/JSON/Markdown.
- Smoke test en `.venv`: PASS.
