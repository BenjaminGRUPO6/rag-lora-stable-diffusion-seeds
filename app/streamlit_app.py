from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.components.demo_helpers import (
    DISCLAIMER,
    ImageValidationError,
    build_download_payload,
    build_markdown_report,
    is_memory_error,
    load_lora_evidence,
    sanitize_error_message,
    source_rows,
    top_probabilities,
    validate_uploaded_image,
)
from src.pipelines.analyze_seed import (
    DEFAULT_INDEX_DIR,
    DEFAULT_RAG_CONFIG,
    DEFAULT_VISION_CONFIG,
    build_available_retriever,
    default_checkpoint_path,
    get_nested,
    load_yaml_config,
    resolve_device,
)
from src.pipelines.analyze_seed import analyze_seed
from src.rag.embeddings import TextEmbedder
from src.rag.retrieval import Retriever
from src.rag.vector_store import FaissStore
from src.vision.inference import VisionInferenceEngine
from src.vision.preprocessing import PreprocessingConfig, PreprocessingResult, preprocess_image


@st.cache_resource(show_spinner="Cargando modelo ResNet18...")
def load_resnet18_resource(
    checkpoint_path: str,
    vision_config_path: str,
    device_name: str | None,
) -> tuple[VisionInferenceEngine, str]:
    """Load and cache the shared ResNet18 inference engine."""
    config = load_yaml_config(vision_config_path)
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint.name}")
    device = resolve_device(device_name)
    engine = VisionInferenceEngine.from_checkpoint(
        checkpoint_path=checkpoint,
        device=device,
        config=config,
    )
    return engine, str(device)


@st.cache_resource(show_spinner="Cargando indice FAISS...")
def load_faiss_index_resource(index_dir: str) -> FaissStore:
    """Load and cache the FAISS index and metadata."""
    root = Path(index_dir)
    index_path = root / "index.faiss"
    metadata_path = root / "metadata.json"
    missing = [path.name for path in (index_path, metadata_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Indice FAISS incompleto. Falta: {', '.join(missing)}")
    return FaissStore.load(index_path, metadata_path)


@st.cache_resource(show_spinner="Cargando embeddings...")
def load_embeddings_resource(model_name: str) -> TextEmbedder:
    """Load and cache the sentence-transformer embeddings model."""
    return TextEmbedder(model_name)


def reset_form() -> None:
    """Clear Streamlit form state and previous analysis output."""
    st.session_state["uploader_version"] = int(st.session_state.get("uploader_version", 0)) + 1
    st.session_state["observations"] = ""
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("report_payload", None)
    st.session_state.pop("preview_image", None)
    st.session_state.pop("preview_original", None)
    st.session_state.pop("preview_preprocessing", None)
    st.session_state.pop("image_input_choice", None)


def build_cached_retriever(rag_config: dict[str, Any]) -> tuple[Retriever | None, int]:
    """Build the best available local retriever."""
    top_k = int(get_nested(rag_config, ("rag", "top_k"), 5))
    retriever, _, _ = build_available_retriever(
        rag_config=rag_config,
        index_dir=DEFAULT_INDEX_DIR,
        top_k=top_k,
    )
    return retriever, top_k


def run_analysis(image: Any, observations: str) -> dict[str, Any]:
    """Run the SeedCare-RAG pipeline with Streamlit-cached resources."""
    vision_config = load_yaml_config(DEFAULT_VISION_CONFIG)
    rag_config = load_yaml_config(DEFAULT_RAG_CONFIG)
    checkpoint = default_checkpoint_path(vision_config)
    engine, device_name = load_resnet18_resource(
        str(checkpoint),
        str(DEFAULT_VISION_CONFIG),
        None,
    )
    retriever, top_k = build_cached_retriever(rag_config)
    return analyze_seed(
        image=image,
        vision_config_path=DEFAULT_VISION_CONFIG,
        rag_config_path=DEFAULT_RAG_CONFIG,
        index_dir=DEFAULT_INDEX_DIR,
        observations=observations,
        inference_engine=engine,
        retriever=retriever,
        top_k=top_k,
        device_name=device_name,
    )


def render_preprocessing_preview(
    original_image: Any,
    preprocessing: PreprocessingResult,
) -> None:
    """Render original image, automatic crop and visual quality controls."""
    st.subheader("Preprocesamiento visual")
    col_original, col_crop = st.columns(2)
    col_original.image(original_image, caption="Imagen original", width=280)
    crop_caption = "Recorte automatico"
    if preprocessing.used_fallback:
        crop_caption = f"Recorte automatico (fallback: {preprocessing.fallback_reason})"
    col_crop.image(preprocessing.crop, caption=crop_caption, width=280)

    selected = st.radio(
        "Imagen para inferencia",
        ["Usar imagen original", "Usar recorte automatico"],
        index=0 if st.session_state.get("image_input_choice") != "crop" else 1,
        horizontal=True,
    )
    st.session_state["image_input_choice"] = (
        "crop" if selected == "Usar recorte automatico" else "original"
    )
    st.session_state["preview_image"] = (
        preprocessing.crop
        if st.session_state["image_input_choice"] == "crop"
        else original_image
    )

    quality = preprocessing.quality
    metric_cols = st.columns(3)
    metric_cols[0].metric("Blur score", f"{quality.blur_score:.2f}")
    metric_cols[1].metric("Brillo", f"{quality.brightness_score:.3f}")
    metric_cols[2].metric("Contraste", f"{quality.contrast_score:.3f}")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Foreground", f"{quality.foreground_ratio:.3f}")
    metric_cols[1].metric("Componentes", str(quality.component_count))
    metric_cols[2].metric("Confianza crop", f"{quality.crop_confidence:.3f}")
    if quality.warnings:
        st.caption("Warnings de control visual")
        st.dataframe(
            pd.DataFrame({"warning": quality.warnings}),
            hide_index=True,
            width="stretch",
        )


def render_prediction(result: dict[str, Any]) -> None:
    """Render prediction, probabilities and uncertainty state."""
    st.subheader("Resultado visual")
    col_label, col_confidence, col_status = st.columns(3)
    col_label.metric("Clase estimada", str(result.get("prediction") or ""))
    col_confidence.metric("Confianza calibrada", f"{float(result.get('confidence') or 0.0):.4f}")
    uncertainty_status = str(result.get("uncertainty_status") or "")
    reliability_status = str(result.get("reliability_status") or "")
    col_status.metric("Estado", reliability_status or uncertainty_status)
    if uncertainty_status == "uncertain":
        st.warning(
            "Prediccion incierta: la confianza es baja o el margen entre clases es estrecho."
        )

    detail_cols = st.columns(2)
    detail_cols[0].metric("Segunda clase", str(result.get("second_class") or "No disponible"))
    detail_cols[1].metric("Margen top1-top2", f"{float(result.get('top1_top2_margin') or 0.0):.4f}")

    top_three = top_probabilities(result.get("probabilities") or {})
    if top_three:
        st.caption("Top 3 probabilidades calibradas")
        st.dataframe(pd.DataFrame(top_three), hide_index=True, width="stretch")

    with st.expander("Seccion tecnica", expanded=False):
        st.metric(
            "Confianza sin calibrar",
            f"{float(result.get('uncalibrated_confidence') or 0.0):.4f}",
        )
        temperature = result.get("calibration_temperature")
        st.metric(
            "Temperatura",
            "No disponible" if temperature is None else f"{float(temperature):.4f}",
        )
        uncalibrated = top_probabilities(result.get("uncalibrated_probabilities") or {})
        if uncalibrated:
            st.caption("Top 3 probabilidades sin calibrar")
            st.dataframe(pd.DataFrame(uncalibrated), hide_index=True, width="stretch")


def render_retrieval(result: dict[str, Any]) -> None:
    """Render retrieved evidence and source metadata."""
    st.subheader("Informacion recuperada")
    rows = source_rows(result.get("retrieved_sources") or [])
    if not rows:
        st.warning("No se recuperaron documentos para sustentar informacion tecnica.")
        return

    for index, row in enumerate(rows, start=1):
        page = row.get("page")
        page_text = f"pagina {page}" if page not in (None, "") else "pagina no indicada"
        with st.expander(f"{index}. {row['title']} ({page_text})", expanded=index == 1):
            st.write(row.get("fragment") or "Sin fragmento disponible.")
            if row.get("url"):
                st.link_button("Abrir fuente", str(row["url"]))
            else:
                st.caption("URL no disponible en la metadata recuperada.")

    st.caption("Fuentes")
    st.dataframe(
        pd.DataFrame(
            [
                {"titulo": row["title"], "pagina": row["page"], "url": row["url"] or "No disponible"}
                for row in rows
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def render_report(result: dict[str, Any]) -> None:
    """Render deterministic preliminary report, limitations and processing times."""
    st.subheader("Informe preliminar")
    report = result.get("preliminary_report") or {}
    summary = report.get("resumen_visual") or report.get("informe_generado")
    if summary:
        st.write(summary)
    else:
        st.info("No hay resumen textual disponible.")

    evidence = report.get("informacion_documental") or []
    if evidence:
        st.caption("Extractos documentales usados")
        for item in evidence:
            title = item.get("title", "Sin titulo") if isinstance(item, dict) else "Fuente"
            fragment = item.get("fragment", "") if isinstance(item, dict) else str(item)
            st.markdown(f"**{title}**")
            st.write(fragment)

    st.subheader("Limitaciones")
    for limitation in result.get("limitations") or []:
        st.write(f"- {limitation}")

    st.subheader("Tiempos de procesamiento")
    times = result.get("processing_times") or {}
    if times:
        st.dataframe(
            pd.DataFrame(
                [{"etapa": key, "segundos": round(float(value), 6)} for key, value in times.items()]
            ),
            hide_index=True,
            width="stretch",
        )


def render_downloads(payload: dict[str, Any]) -> None:
    """Render JSON and Markdown report download buttons."""
    st.subheader("Descargas")
    markdown_report = build_markdown_report(payload)
    col_json, col_markdown = st.columns(2)
    col_json.download_button(
        "Descargar reporte JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name="seedcare_rag_report.json",
        mime="application/json",
    )
    col_markdown.download_button(
        "Descargar reporte Markdown",
        data=markdown_report,
        file_name="seedcare_rag_report.md",
        mime="text/markdown",
    )


def render_lora_section() -> None:
    """Render optional selected LoRA evidence from local result files."""
    with st.expander("Acerca del modelo generativo LoRA", expanded=False):
        evidence = load_lora_evidence(Path("results/lora"))
        if not evidence.get("available"):
            st.info(str(evidence.get("message") or "No hay evidencia LoRA disponible."))
            return

        st.caption("Evidencia local seleccionada. No se muestran pesos ni rutas privadas.")
        col_status, col_dataset = st.columns(2)
        col_status.metric("Estado", str(evidence.get("status") or ""))
        dataset_images = evidence.get("dataset_images")
        col_dataset.metric("Registros de metadata", str(dataset_images or "No indicado"))

        if evidence.get("no_retraining_performed"):
            st.info("No se ejecuto reentrenamiento durante esta demostracion.")

        class_distribution = evidence.get("class_distribution") or {}
        if class_distribution:
            st.caption("Distribucion por categoria visual")
            st.dataframe(
                pd.DataFrame(
                    [{"categoria": key, "cantidad": value} for key, value in class_distribution.items()]
                ),
                hide_index=True,
                width="stretch",
            )

        parameters = evidence.get("parameters") or {}
        if parameters:
            st.caption("Parametros confirmados")
            st.dataframe(
                pd.DataFrame(
                    [{"parametro": key, "valor": str(value)} for key, value in parameters.items()]
                ),
                hide_index=True,
                width="stretch",
            )

        missing_evidence = evidence.get("missing_evidence") or []
        if missing_evidence:
            st.caption("Evidencia faltante")
            for item in missing_evidence:
                st.write(f"- {item}")

        notes = evidence.get("notes") or []
        for note in notes:
            st.write(f"- {note}")

        sample_names = evidence.get("sample_names") or []
        if sample_names:
            st.caption("Muestras disponibles en results/lora")
            sample_paths = [Path("results/lora/samples") / name for name in sample_names[:6]]
            st.image(sample_paths, caption=sample_names[:6], width=160)


def render_error(exc: BaseException) -> None:
    """Render known error classes without exposing private local paths."""
    message = sanitize_error_message(str(exc), REPO_ROOT)
    if isinstance(exc, FileNotFoundError):
        st.error(f"Recurso faltante: {message}")
    elif is_memory_error(exc):
        st.error("No se pudo completar el analisis por falta de memoria.")
        st.caption(message)
    else:
        st.error(f"No se pudo completar el analisis ({exc.__class__.__name__}).")
        if message:
            st.caption(message)


st.set_page_config(page_title="SeedCare-RAG", layout="wide")
st.title("SeedCare-RAG")
st.caption("Clasificacion visual preliminar, recuperacion tecnica e informe con fuentes.")
st.warning(DISCLAIMER)

if "uploader_version" not in st.session_state:
    st.session_state["uploader_version"] = 0
if "observations" not in st.session_state:
    st.session_state["observations"] = ""

left, right = st.columns([0.42, 0.58])
with left:
    uploaded = st.file_uploader(
        "Carga una imagen de semilla de soja",
        type=["jpg", "jpeg", "png"],
        key=f"seed_image_{st.session_state['uploader_version']}",
    )
    observations = st.text_area("Observaciones opcionales", key="observations", height=120)
    action_col, clear_col = st.columns(2)
    run_clicked = action_col.button("Ejecutar analisis", type="primary", width="stretch")
    clear_col.button("Limpiar formulario", width="stretch", on_click=reset_form)

with right:
    if uploaded is None:
        st.info("Carga un JPG, JPEG o PNG para iniciar.")
    else:
        try:
            image = validate_uploaded_image(uploaded.name, uploaded.getvalue())
            preprocessing = preprocess_image(
                image,
                config=PreprocessingConfig(
                    output_size=int(
                        get_nested(load_yaml_config(DEFAULT_VISION_CONFIG), ("data", "image_size"), 224)
                    )
                ),
            )
            st.session_state["preview_original"] = image
            st.session_state["preview_preprocessing"] = preprocessing
            render_preprocessing_preview(image, preprocessing)
        except ImageValidationError as exc:
            st.session_state.pop("preview_image", None)
            st.session_state.pop("preview_original", None)
            st.session_state.pop("preview_preprocessing", None)
            st.error(str(exc))

if run_clicked:
    image = st.session_state.get("preview_image")
    if image is None:
        st.error("Carga una imagen valida antes de ejecutar el analisis.")
    else:
        try:
            with st.spinner("Ejecutando analisis SeedCare-RAG..."):
                result = run_analysis(image, st.session_state.get("observations", ""))
            payload = build_download_payload(result, st.session_state.get("observations", ""))
            st.session_state["analysis_result"] = result
            st.session_state["report_payload"] = payload
        except Exception as exc:
            render_error(exc)

result = st.session_state.get("analysis_result")
payload = st.session_state.get("report_payload")
if result:
    tab_result, tab_sources, tab_report, tab_downloads = st.tabs(
        ["Resultado", "Evidencia RAG", "Informe", "Descargas"]
    )
    with tab_result:
        render_prediction(result)
    with tab_sources:
        render_retrieval(result)
    with tab_report:
        render_report(result)
    with tab_downloads:
        if payload:
            render_downloads(payload)

render_lora_section()
