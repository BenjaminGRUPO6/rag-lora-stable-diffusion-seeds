from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(page_title="SeedCare-RAG LoRA", layout="wide")
st.title("SeedCare-RAG LoRA")
st.caption("Análisis preliminar de daños en semillas con recuperación de fuentes técnicas")

uploaded = st.file_uploader("Cargue una imagen de semilla", type=["jpg", "jpeg", "png", "webp"])
observations = st.text_area("Observaciones opcionales", placeholder="Ej.: manchas oscuras, perforación o deformación")

left, right = st.columns(2)
with left:
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Imagen recibida", use_container_width=True)
    else:
        st.info("Cargue una imagen para iniciar.")

with right:
    st.subheader("Resultado")
    st.warning("Plantilla inicial: conecte el modelo visual, el RAG y el generador de informes.")
    st.markdown(
        """
        **Salida prevista:**
        - clase estimada y confianza;
        - características observadas;
        - fuentes recuperadas;
        - posibles causas y métodos de prevención/manejo;
        - limitaciones y recomendación de revisión especializada.
        """
    )

st.divider()
st.subheader("Módulo de entrenamiento")
st.markdown(
    "El proyecto entrena un clasificador visual y un adaptador LoRA de Stable Diffusion 1.5. "
    "Las imágenes sintéticas aceptadas se usan solo en el conjunto de entrenamiento."
)
