from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="SeedCare-RAG LoRA", layout="wide")
st.title("SeedCare-RAG LoRA")
st.caption("Clasificación visual preliminar, recuperación técnica e informe con fuentes")

uploaded = st.file_uploader("Carga una imagen de una semilla de soja", type=["jpg", "jpeg", "png", "webp"])
observations = st.text_area("Observaciones opcionales")

if uploaded:
    st.image(uploaded, caption="Imagen recibida", width=320)
    st.info("La interfaz está preparada. Falta integrar el clasificador entrenado y el RAG.")

st.warning("El resultado final será preliminar y no sustituirá una evaluación fitosanitaria especializada.")
