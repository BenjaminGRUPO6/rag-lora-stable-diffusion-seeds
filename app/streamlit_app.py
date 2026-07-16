from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
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
    sanitize_error_message,
    source_rows,
    top_probabilities,
    validate_uploaded_image,
)
from src.synthetic_data.lora_evidence import (
    MANDATORY_EXPLANATION,
    VISION_LORA_RESULTS_DIR,
    load_lora_visual_evidence,
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
from src.vision.gradcam import find_last_convolutional_layer, generate_gradcam_with_fallback
from src.vision.inference import VisionInferenceEngine
from src.vision.preprocessing import PreprocessingConfig, PreprocessingResult, preprocess_image
from src.vision.visualization import (
    build_combined_gradcam_image,
    heatmap_to_image,
    image_to_png_bytes,
)

APP_PRODUCTION_CONFIG = Path("configs/production_vision_model.yaml")
APP_VISION_CONFIG = Path("configs/vision_v2_resnet18.yaml")
APP_EFFICIENTNET_CONFIG = Path("configs/vision_v2_efficientnet_b0.yaml")
TTA_SELECTION_PATH = Path("results/vision/resultados_2_mejoras/07_tta/selected_tta_policy.json")
GRADCAM_RESULTS_DIR = Path("results/vision/resultados_2_mejoras/09_gradcam_interfaz")
RESULTS_1_DIR = Path("results/vision/resultados_1_baseline")
RESULTS_2_DIR = Path("results/vision/resultados_2_mejoras")
FINAL_RESULTS_DIR = RESULTS_2_DIR / "final"


@st.cache_resource(show_spinner="Cargando modelo de vision...")
def load_vision_resource(
    checkpoint_path: str,
    vision_config_path: str,
    production_config_path: str,
    device_name: str | None,
) -> tuple[VisionInferenceEngine, str, dict[str, Any], str]:
    """Load and cache the production vision inference engine."""
    config = load_yaml_config(vision_config_path)
    production_config = load_yaml_config(production_config_path)
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint.name}")
    device = resolve_device(device_name)
    temperature_path = production_config.get("calibration_path")
    engine = VisionInferenceEngine.from_checkpoint(
        checkpoint_path=checkpoint,
        device=device,
        config=config,
        temperature_path=temperature_path,
    )
    try:
        gradcam_layer, _ = find_last_convolutional_layer(engine.model)
    except ValueError:
        gradcam_layer = "fallback"
    return engine, str(device), production_config, gradcam_layer


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


def app_checkpoint_path(vision_config: dict[str, Any]) -> Path:
    """Return the checkpoint path used by the Streamlit app."""
    configured = get_nested(vision_config, ("output", "checkpoint_path"), None)
    if configured:
        return Path(str(configured))
    return default_checkpoint_path(vision_config)


def selected_vision_config_path(production_config: dict[str, Any]) -> Path:
    """Return the detailed vision config for the selected production architecture."""
    architecture = str(production_config.get("architecture") or "resnet18").lower()
    if architecture == "efficientnet_b0":
        return APP_EFFICIENTNET_CONFIG
    return APP_VISION_CONFIG


def production_checkpoint_path(production_config: dict[str, Any], vision_config: dict[str, Any]) -> Path:
    """Return the checkpoint path selected for production inference."""
    configured = production_config.get("checkpoint_path")
    if configured:
        return Path(str(configured))
    return app_checkpoint_path(vision_config)


def load_production_selection() -> tuple[Path, dict[str, Any], Path]:
    """Load production model settings and resolve its detailed config plus checkpoint."""
    production_config = load_yaml_config(APP_PRODUCTION_CONFIG)
    vision_config_path = selected_vision_config_path(production_config)
    vision_config = load_yaml_config(vision_config_path)
    checkpoint = production_checkpoint_path(production_config, vision_config)
    return vision_config_path, production_config, checkpoint


def load_tta_selection(path: Path = TTA_SELECTION_PATH) -> dict[str, Any]:
    """Load the validation-selected TTA policy for optional app inference."""
    if not path.exists():
        return {
            "tta_enabled": False,
            "default_enabled": False,
            "selected_policy": "none",
            "views": 1,
            "temperature": None,
            "reason": "TTA no evaluado.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Archivo TTA invalido: {path}")
    return payload


def build_cached_retriever(rag_config: dict[str, Any]) -> tuple[Retriever | None, int]:
    """Build the best available local retriever."""
    top_k = int(get_nested(rag_config, ("rag", "top_k"), 5))
    retriever, _, _ = build_available_retriever(
        rag_config=rag_config,
        index_dir=DEFAULT_INDEX_DIR,
        top_k=top_k,
    )
    return retriever, top_k


def run_analysis(image: Any, observations: str, *, use_tta: bool) -> dict[str, Any]:
    """Run the SeedCare-RAG pipeline with Streamlit-cached resources."""
    vision_config_path, _, checkpoint = load_production_selection()
    rag_config = load_yaml_config(DEFAULT_RAG_CONFIG)
    tta_selection = load_tta_selection()
    selected_policy = str(tta_selection.get("selected_policy") or "none")
    selected_temperature = tta_selection.get("temperature")
    tta_allowed = bool(tta_selection.get("tta_enabled")) and selected_policy != "none"
    engine, device_name, _, _ = load_vision_resource(
        str(checkpoint),
        str(vision_config_path),
        str(APP_PRODUCTION_CONFIG),
        None,
    )
    retriever, top_k = build_cached_retriever(rag_config)
    return analyze_seed(
        image=image,
        vision_config_path=vision_config_path,
        rag_config_path=DEFAULT_RAG_CONFIG,
        index_dir=DEFAULT_INDEX_DIR,
        observations=observations,
        inference_engine=engine,
        retriever=retriever,
        top_k=top_k,
        device_name=device_name,
        use_tta=bool(use_tta and tta_allowed),
        tta_policy_name=selected_policy,
        tta_temperature=float(selected_temperature) if selected_temperature is not None else None,
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
    if result.get("tta_enabled"):
        tta_cols = st.columns(3)
        tta_cols[0].metric("TTA", str(result.get("tta_policy") or "none"))
        tta_cols[1].metric("Vistas", str(result.get("tta_views") or 1))
        tta_cols[2].metric(
            "Tiempo adicional",
            f"{float(result.get('tta_extra_seconds') or 0.0):.4f} s",
        )
    else:
        st.caption("TTA desactivado para este analisis.")

    top_three = top_probabilities(result.get("probabilities") or {})
    if top_three:
        st.caption("Top 3 probabilidades calibradas")
        st.dataframe(pd.DataFrame(top_three), hide_index=True, width="stretch")
    render_probability_chart(result.get("probabilities") or {})

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


def render_probability_chart(probabilities: dict[str, float]) -> None:
    """Render a horizontal probability chart."""
    if not probabilities:
        return
    frame = pd.DataFrame(
        [
            {"clase": str(label), "probabilidad": float(value)}
            for label, value in probabilities.items()
        ]
    ).sort_values("probabilidad", ascending=True)
    figure = px.bar(
        frame,
        x="probabilidad",
        y="clase",
        orientation="h",
        range_x=[0, 1],
        text=frame["probabilidad"].map(lambda value: f"{value:.3f}"),
    )
    figure.update_layout(
        height=260,
        margin={"l": 12, "r": 12, "t": 18, "b": 18},
        xaxis_title="Probabilidad",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(figure, use_container_width=True)


def build_gradcam_display(
    *,
    original_image: Any,
    crop_image: Any,
    inference_image: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Generate Grad-CAM display artifacts with a safe fallback."""
    vision_config_path, _, checkpoint = load_production_selection()
    engine, _, production_config, production_layer = load_vision_resource(
        str(checkpoint),
        str(vision_config_path),
        str(APP_PRODUCTION_CONFIG),
        None,
    )
    prediction = str(result.get("prediction") or "")
    target_index = engine.labels.index(prediction) if prediction in engine.labels else None
    gradcam = generate_gradcam_with_fallback(
        model=engine.model,
        image=inference_image,
        transform=engine.transform,
        device=engine.device,
        target_class_index=target_index,
    )
    heatmap = heatmap_to_image(gradcam.heatmap, inference_image.size)
    combined = build_combined_gradcam_image(
        original=original_image,
        crop=crop_image,
        heatmap=heatmap,
        overlay=gradcam.overlay,
        title="SeedCare-RAG - panel visual",
        metadata={
            "modelo": str(production_config.get("model_name") or production_config.get("architecture") or ""),
            "clase": prediction,
            "confianza": f"{float(result.get('confidence') or 0.0):.3f}",
            "capa": gradcam.target_layer_name,
        },
    )
    return {
        "heatmap": heatmap,
        "overlay": gradcam.overlay,
        "combined": combined,
        "status": gradcam.status,
        "message": gradcam.message,
        "intensity": gradcam.intensity,
        "layer": gradcam.target_layer_name,
        "production_layer": production_layer,
    }


def render_gradcam(gradcam_display: dict[str, Any] | None) -> None:
    """Render Grad-CAM heatmap, overlay and fallback state."""
    st.subheader("Explicabilidad visual")
    st.info("Grad-CAM es una explicacion aproximada del modelo; no es prueba causal ni diagnostico.")
    if not gradcam_display:
        st.caption("Ejecuta un analisis para generar Grad-CAM.")
        render_static_gradcam_graphs()
        return
    col_heatmap, col_overlay = st.columns(2)
    col_heatmap.image(gradcam_display["heatmap"], caption="Heatmap Grad-CAM", width=280)
    col_overlay.image(gradcam_display["overlay"], caption="Overlay Grad-CAM", width=280)
    metric_cols = st.columns(3)
    metric_cols[0].metric("Intensidad normalizada", f"{float(gradcam_display['intensity']):.4f}")
    metric_cols[1].metric("Capa", str(gradcam_display["layer"]))
    metric_cols[2].metric("Estado", str(gradcam_display["status"]))
    if gradcam_display.get("status") != "ok":
        st.warning(str(gradcam_display.get("message") or "Grad-CAM no disponible."))
    st.image(gradcam_display["combined"], caption="Imagen combinada", width=900)


def render_static_gradcam_graphs() -> None:
    """Render generated Grad-CAM evidence PNGs when available."""
    pngs = [
        "r2_gradcam_correctos.png",
        "r2_gradcam_errores.png",
        "r2_gradcam_intact_broken.png",
        "r2_panel_visual_demo.png",
    ]
    available = [GRADCAM_RESULTS_DIR / name for name in pngs if (GRADCAM_RESULTS_DIR / name).exists()]
    if not available:
        st.caption("Aun no hay PNG Grad-CAM generados localmente.")
        return
    for path in available:
        st.image(path, caption=path.name, width=900)


def render_results_comparison() -> None:
    """Render Resultados 1, Resultados 2 and comparison PNGs."""
    final_metrics_path = FINAL_RESULTS_DIR / "final_metrics.json"
    if final_metrics_path.exists():
        final_metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))
        final_model = final_metrics.get("final_model", {})
        improvement = final_metrics.get("improvement_vs_resultados_1", {}).get("test_macro_f1", {})
        st.subheader("Configuracion final de produccion")
        cols = st.columns(4)
        cols[0].metric("Modelo", str(final_model.get("model_id", "N/D")))
        cols[1].metric("Validation macro-F1", f"{float(final_model.get('validation_macro_f1', 0.0)):.6f}")
        cols[2].metric("Test macro-F1", f"{float(final_model.get('test_macro_f1', 0.0)):.6f}")
        cols[3].metric(
            "Mejora test vs R1",
            f"{float(improvement.get('absolute', 0.0)):.6f}",
            f"{float(improvement.get('percent', 0.0)):.2f}%",
        )
        st.caption("Seleccion por validation; test se muestra solo como evaluacion final.")
        comparison_path = FINAL_RESULTS_DIR / "final_comparison.csv"
        if comparison_path.exists():
            st.dataframe(pd.read_csv(comparison_path), hide_index=True, width="stretch")
        render_png_gallery(
            [
                FINAL_RESULTS_DIR / "r1_vs_r2_dashboard.png",
                FINAL_RESULTS_DIR / "r1_vs_r2_metricas_globales.png",
                FINAL_RESULTS_DIR / "r1_vs_r2_f1_por_clase.png",
                FINAL_RESULTS_DIR / "r1_vs_r2_confianza.png",
                FINAL_RESULTS_DIR / "r1_vs_r2_latencia.png",
                FINAL_RESULTS_DIR / "r2_sistema_final.png",
            ]
        )
    st.subheader("Resultados 1")
    render_png_gallery(
        [
            RESULTS_1_DIR / "r1_metricas_resumen.png",
            RESULTS_1_DIR / "r1_matriz_confusion.png",
            RESULTS_1_DIR / "r1_f1_por_clase.png",
        ]
    )
    st.subheader("Resultados 2")
    render_png_gallery(
        [
            RESULTS_2_DIR / "05_resnet18_v2" / "r2_matriz_confusion_resnet18_v2.png",
            RESULTS_2_DIR / "05_resnet18_v2" / "r2_f1_por_clase_resnet18_v2.png",
            RESULTS_2_DIR / "08_comparacion_modelos" / "r2_efficientnet_confusion_matrix.png",
            RESULTS_2_DIR / "08_comparacion_modelos" / "r2_resnet18_vs_efficientnet_f1.png",
        ]
    )
    st.subheader("Comparacion")
    render_png_gallery(
        [
            GRADCAM_RESULTS_DIR / "r1_vs_r2_dashboard.png",
            RESULTS_2_DIR / "08_comparacion_modelos" / "r1_vs_r2_modelos_dashboard.png",
            RESULTS_2_DIR / "05_resnet18_v2" / "r1_vs_r2_f1_resnet18.png",
        ]
    )


def render_png_gallery(paths: list[Path]) -> None:
    """Render existing PNG paths without failing when optional files are absent."""
    existing = [path for path in paths if path.exists()]
    if not existing:
        st.caption("No hay graficos PNG disponibles en esta seccion.")
        return
    for path in existing:
        st.image(path, caption=path.name, width=760)


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


def render_downloads(payload: dict[str, Any], panel_image: Any | None = None) -> None:
    """Render PNG, JSON and Markdown report download buttons."""
    st.subheader("Descargas")
    markdown_report = build_markdown_report(payload)
    col_png, col_json, col_markdown = st.columns(3)
    if panel_image is not None:
        col_png.download_button(
            "Descargar panel PNG",
            data=image_to_png_bytes(panel_image),
            file_name="seedcare_rag_panel.png",
            mime="image/png",
        )
    else:
        col_png.caption("Panel PNG disponible despues del analisis.")
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
    """Render the separate LoRA generative evidence section."""
    st.subheader("Modelo generativo LoRA")
    evidence = load_lora_visual_evidence()
    if not evidence.get("available"):
        st.info(str(evidence.get("message") or "No hay evidencia LoRA consolidada."))
        return

    st.info(MANDATORY_EXPLANATION)
    st.warning(
        "Las imagenes sinteticas solo pueden incorporarse a train despues de revision humana."
    )

    status_col, dataset_col, comparison_col = st.columns(3)
    status_col.metric("Estado evidencia", str(evidence.get("status") or ""))
    dataset = evidence.get("dataset") or {}
    dataset_col.metric("Imagenes metadata", str(dataset.get("referenced_images_existing") or 0))
    comparison_col.metric(
        "Base vs adaptado",
        str((evidence.get("comparison") or {}).get("status") or "EVIDENCE_MISSING"),
    )

    st.subheader("Que hace")
    st.write(
        "- Genera imagenes sinteticas de semillas cuando el adaptador se carga "
        "externamente en un pipeline Stable Diffusion 1.5."
    )
    st.write("- Documenta evidencia local ya existente del entrenamiento LoRA.")

    st.subheader("Que no hace")
    st.write("- No clasifica la imagen cargada en esta aplicacion.")
    st.write("- No modifica la confianza ni las probabilidades del clasificador ResNet18.")
    st.write("- No carga Stable Diffusion ni el safetensors durante el inicio normal.")
    st.write("- No ejecuta generacion masiva ni reentrenamiento.")

    model = evidence.get("model") or {}
    adapter = model.get("adapter") if isinstance(model.get("adapter"), dict) else {}
    parameters = (evidence.get("training") or {}).get("parameters") or {}
    parameter_rows = []
    for key in [
        "base_model",
        "trigger_word",
        "resolution",
        "rank",
        "learning_rate",
        "max_train_steps_initial",
        "max_train_steps_full",
        "train_batch_size",
        "gradient_accumulation_steps",
        "mixed_precision",
        "seed",
    ]:
        value = parameters.get(key)
        if isinstance(value, dict):
            parameter_rows.append(
                {
                    "parametro": key,
                    "valor": str(value.get("value")),
                    "fuente": str(value.get("source")),
                }
            )
    parameter_rows.append(
        {"parametro": "hardware", "valor": "EVIDENCE_MISSING", "fuente": "EVIDENCE_MISSING"}
    )
    if adapter:
        parameter_rows.append(
            {
                "parametro": "adapter_name",
                "valor": str(adapter.get("adapter_name") or "EVIDENCE_MISSING"),
                "fuente": "models/lora",
            }
        )
    st.subheader("Modelo base y entrenamiento")
    st.dataframe(pd.DataFrame(parameter_rows), hide_index=True, width="stretch")

    class_distribution = dataset.get("class_distribution") or {}
    if class_distribution:
        st.subheader("Clases visuales")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "clase": key,
                        "cantidad": value,
                        "nota": "categoria visual; no diagnostico de hongo"
                        if key == "spotted"
                        else "",
                    }
                    for key, value in class_distribution.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Galeria")
    sample_rows = evidence.get("samples") or []
    sample_paths = [
        REPO_ROOT / str(row.get("image_path"))
        for row in sample_rows
        if row.get("image_path") and (REPO_ROOT / str(row.get("image_path"))).exists()
    ]
    if sample_paths:
        st.image(
            sample_paths[:5],
            caption=[path.name for path in sample_paths[:5]],
            width=160,
        )
        st.caption("Muestras existentes de metadata de entrenamiento; no son generacion nueva.")
    else:
        st.info("No hay imagenes de muestra verificadas para mostrar.")

    prompt_rows = [
        {
            "clase": row.get("class_name", ""),
            "prompt_o_caption": row.get("prompt", "") or "EVIDENCE_MISSING",
            "seed": row.get("seed", "") or "EVIDENCE_MISSING",
            "estado": row.get("evidence_status", ""),
        }
        for row in sample_rows
    ]
    if prompt_rows:
        st.subheader("Prompt y seed")
        st.dataframe(pd.DataFrame(prompt_rows), hide_index=True, width="stretch")

    st.subheader("Evidencia visual generada")
    render_png_gallery(
        [
            VISION_LORA_RESULTS_DIR / "r2_lora_model_card.png",
            VISION_LORA_RESULTS_DIR / "r2_lora_base_vs_adaptado.png",
            VISION_LORA_RESULTS_DIR / "r2_lora_clases.png",
            VISION_LORA_RESULTS_DIR / "r2_lora_flujo.png",
        ]
    )

    missing = evidence.get("evidence_missing") or []
    if missing:
        st.subheader("Limitaciones y evidencia faltante")
        for item in missing:
            st.write(f"- {item}")
    for limitation in evidence.get("limitations") or []:
        st.write(f"- {limitation}")


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
st.caption("Clasificacion visual preliminar, Grad-CAM aproximado, recuperacion tecnica e informe con fuentes.")
st.warning(DISCLAIMER)

if "uploader_version" not in st.session_state:
    st.session_state["uploader_version"] = 0
if "observations" not in st.session_state:
    st.session_state["observations"] = ""

tta_selection = load_tta_selection()
tta_available = bool(tta_selection.get("tta_enabled")) and str(
    tta_selection.get("selected_policy") or "none"
) != "none"

_LEGACY_LAYOUT_FOR_DIFF = """Legacy single-page layout retained only for diff context.
left, right = st.columns([0.42, 0.58])
with left:
    uploaded = st.file_uploader(
        "Carga una imagen de semilla de soja",
        type=["jpg", "jpeg", "png"],
        key=f"seed_image_{st.session_state['uploader_version']}",
    )
    observations = st.text_area("Observaciones opcionales", key="observations", height=120)
    use_tta = st.toggle(
        "Análisis estable con TTA",
        value=bool(tta_available and tta_selection.get("default_enabled", False)),
        disabled=not tta_available,
        help=str(tta_selection.get("reason") or ""),
    )
    st.caption(
        "Vistas TTA seleccionadas: "
        f"{int(tta_selection.get('views') or 1)}; politica: "
        f"{str(tta_selection.get('selected_policy') or 'none')}."
    )
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
                result = run_analysis(
                    image,
                    st.session_state.get("observations", ""),
                    use_tta=use_tta,
                )
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
"""

tab_analysis, tab_gradcam, tab_rag, tab_results, tab_lora = st.tabs(
    [
        "A. Análisis",
        "B. Explicabilidad",
        "C. Evidencia RAG",
        "D. Resultados 1 vs Resultados 2",
        "Modelo generativo LoRA",
    ]
)

with tab_analysis:
    controls, preview = st.columns([0.34, 0.66])
    with controls:
        uploaded = st.file_uploader(
            "Carga una imagen de semilla de soja",
            type=["jpg", "jpeg", "png"],
            key=f"seed_image_{st.session_state['uploader_version']}",
        )
        observations = st.text_area("Observaciones opcionales", key="observations", height=120)
        use_tta = st.toggle(
            "Analisis estable con TTA",
            value=bool(tta_available and tta_selection.get("default_enabled", False)),
            disabled=not tta_available,
            help=str(tta_selection.get("reason") or ""),
        )
        st.caption(
            "Vistas TTA seleccionadas: "
            f"{int(tta_selection.get('views') or 1)}; politica: "
            f"{str(tta_selection.get('selected_policy') or 'none')}."
        )
        action_col, clear_col = st.columns(2)
        run_clicked = action_col.button("Ejecutar analisis", type="primary", width="stretch")
        clear_col.button("Limpiar formulario", width="stretch", on_click=reset_form)

    with preview:
        if uploaded is None:
            st.info("Carga un JPG, JPEG o PNG para iniciar.")
        else:
            try:
                image = validate_uploaded_image(uploaded.name, uploaded.getvalue())
                _, _, checkpoint = load_production_selection()
                production_config = load_yaml_config(APP_PRODUCTION_CONFIG)
                default_size = get_nested(load_yaml_config(DEFAULT_VISION_CONFIG), ("data", "image_size"), 224)
                preprocessing = preprocess_image(
                    image,
                    config=PreprocessingConfig(
                        output_size=int(production_config.get("image_size", default_size))
                    ),
                )
                st.session_state["preview_original"] = image
                st.session_state["preview_preprocessing"] = preprocessing
                st.caption(f"Checkpoint seleccionado: {checkpoint.name}")
                render_preprocessing_preview(image, preprocessing)
            except ImageValidationError as exc:
                st.session_state.pop("preview_image", None)
                st.session_state.pop("preview_original", None)
                st.session_state.pop("preview_preprocessing", None)
                st.error(str(exc))

    if run_clicked:
        image = st.session_state.get("preview_image")
        original = st.session_state.get("preview_original")
        preprocessing = st.session_state.get("preview_preprocessing")
        if image is None or original is None or preprocessing is None:
            st.error("Carga una imagen valida antes de ejecutar el analisis.")
        else:
            try:
                with st.spinner("Ejecutando analisis SeedCare-RAG..."):
                    result = run_analysis(
                        image,
                        st.session_state.get("observations", ""),
                        use_tta=use_tta,
                    )
                    gradcam_display = build_gradcam_display(
                        original_image=original,
                        crop_image=preprocessing.crop,
                        inference_image=image,
                        result=result,
                    )
                payload = build_download_payload(result, st.session_state.get("observations", ""))
                st.session_state["analysis_result"] = result
                st.session_state["report_payload"] = payload
                st.session_state["gradcam_display"] = gradcam_display
            except Exception as exc:
                render_error(exc)

    result = st.session_state.get("analysis_result")
    payload = st.session_state.get("report_payload")
    gradcam_display = st.session_state.get("gradcam_display")
    if result:
        original = st.session_state.get("preview_original")
        preprocessing = st.session_state.get("preview_preprocessing")
        if original is not None and preprocessing is not None:
            image_cols = st.columns(3)
            image_cols[0].image(original, caption="Original", width=220)
            image_cols[1].image(preprocessing.crop, caption="Crop", width=220)
            if gradcam_display:
                image_cols[2].image(gradcam_display["overlay"], caption="Grad-CAM", width=220)
        render_prediction(result)
        quality = getattr(preprocessing, "quality", None) if preprocessing is not None else None
        if quality is not None:
            st.subheader("Calidad visual")
            quality_cols = st.columns(4)
            quality_cols[0].metric("Blur", f"{quality.blur_score:.2f}")
            quality_cols[1].metric("Brillo", f"{quality.brightness_score:.3f}")
            quality_cols[2].metric("Contraste", f"{quality.contrast_score:.3f}")
            quality_cols[3].metric("Foreground", f"{quality.foreground_ratio:.3f}")
        st.subheader("Tiempo")
        times = result.get("processing_times") or {}
        if times:
            st.dataframe(
                pd.DataFrame(
                    [{"etapa": key, "segundos": round(float(value), 6)} for key, value in times.items()]
                ),
                hide_index=True,
                width="stretch",
            )
        if payload:
            render_downloads(payload, gradcam_display.get("combined") if gradcam_display else None)

with tab_gradcam:
    render_gradcam(st.session_state.get("gradcam_display"))

with tab_rag:
    result = st.session_state.get("analysis_result")
    if result:
        render_retrieval(result)
        render_report(result)
    else:
        st.info("Ejecuta un analisis para ver evidencia RAG e informe preliminar.")

with tab_results:
    render_results_comparison()

with tab_lora:
    render_lora_section()
