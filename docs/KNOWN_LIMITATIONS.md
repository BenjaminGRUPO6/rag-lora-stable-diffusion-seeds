# Known Limitations

- La clasificacion es visual y preliminar; no constituye diagnostico fitosanitario.
- `spotted` es una categoria visual, no una confirmacion de hongo o enfermedad.
- La recuperacion RAG depende de documentos locales aceptados y del indice existente.
- Algunos textos extraidos desde PDF tienen ruido OCR/extraccion.
- El fallback lexical local es menos preciso que embeddings FAISS, pero no inventa fuentes.
- No se incorporaron datos sinteticos a `train`.
- No se reentreno ningun modelo durante la auditoria.
- El titulo renderizado de la app se valida con `streamlit.testing.v1.AppTest`; una descarga HTTP simple de `/` solo comprueba que el shell de Streamlit responde.
