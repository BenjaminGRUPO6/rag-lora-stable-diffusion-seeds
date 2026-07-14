from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

from src.rag.prompt_builder import build_report_prompt, build_retrieval_query
from src.reports.report_generator import (
    ReportProvider,
    default_limitations,
    generate_preliminary_report,
)
from src.vision.inference import (
    ImageInput,
    VisionInferenceEngine,
    build_inference_transform,
)


DEFAULT_VISION_CONFIG = Path("configs/vision_config.yaml")
DEFAULT_RAG_CONFIG = Path("configs/rag.yaml")
DEFAULT_INDEX_DIR = Path("vector_db")

if TYPE_CHECKING:
    from src.rag.retrieval import FaissRetriever, Retriever


def analyze_seed(
    image: ImageInput | None = None,
    *,
    vision_config_path: str | Path = DEFAULT_VISION_CONFIG,
    rag_config_path: str | Path = DEFAULT_RAG_CONFIG,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    checkpoint_path: str | Path | None = None,
    observations: list[str] | str | None = None,
    top_k: int | None = None,
    device_name: str | None = None,
    model: torch.nn.Module | None = None,
    transform: Callable[[Any], torch.Tensor] | None = None,
    labels: list[str] | None = None,
    inference_engine: VisionInferenceEngine | None = None,
    retriever: Callable[..., list[dict]] | None = None,
    prediction: dict | None = None,
    report_provider: ReportProvider | None = None,
) -> dict:
    """Analyze a soybean seed image with vision inference, RAG and a preliminary report.

    Tests may inject ``prediction`` and ``retriever`` to avoid loading model weights or FAISS.
    Production use should pass ``image`` and allow the function to load the configured assets.
    """
    started_at = time.perf_counter()
    observations_list = normalize_observations(observations)
    vision_config = load_yaml_config(vision_config_path)
    rag_config = load_yaml_config(rag_config_path)
    processing_times: dict[str, float] = {}

    vision_started = time.perf_counter()
    if prediction is None:
        if image is None:
            raise ValueError("Debe proporcionar una imagen o una prediccion simulada.")
        device = resolve_device(device_name)
        active_engine = inference_engine
        if active_engine is None:
            active_checkpoint = checkpoint_path or default_checkpoint_path(vision_config)
            if model is None or labels is None:
                active_engine = VisionInferenceEngine.from_checkpoint(
                    checkpoint_path=active_checkpoint,
                    device=device,
                    config=vision_config,
                )
            else:
                active_engine = VisionInferenceEngine(
                    model=model,
                    labels=labels,
                    transform=transform
                    or build_inference_transform(
                        image_size=int(get_nested(vision_config, ("data", "image_size"), 224))
                    ),
                    device=device,
                )
        prediction = active_engine.predict_dict(image)
    processing_times["vision_seconds"] = elapsed(vision_started)

    normalized_prediction = normalize_prediction(prediction)
    uncertainty_status = compute_uncertainty_status(
        probabilities=normalized_prediction["probabilities"],
        confidence=float(normalized_prediction["confidence"]),
        confidence_threshold=float(
            get_nested(vision_config, ("inference", "confidence_threshold"), 0.60)
        ),
        margin_threshold=float(get_nested(vision_config, ("inference", "margin_threshold"), 0.15)),
    )

    retrieval_started = time.perf_counter()
    active_top_k = top_k or int(get_nested(rag_config, ("rag", "top_k"), 5))
    observations_text = "; ".join(observations_list)
    retrieval_query = build_retrieval_query(
        str(normalized_prediction["label"]),
        observations_text or None,
    )
    rag_status = "injected"
    rag_warning = ""
    if retriever is None:
        active_retriever, rag_status, rag_warning = build_available_retriever(
            rag_config=rag_config,
            index_dir=index_dir,
            top_k=active_top_k,
        )
    else:
        active_retriever = retriever
    retrieved_sources = (
        call_retriever(active_retriever, retrieval_query, active_top_k)
        if active_retriever is not None
        else []
    )
    processing_times["retrieval_seconds"] = elapsed(retrieval_started)

    report_started = time.perf_counter()
    prompt = build_report_prompt(normalized_prediction, retrieved_sources)
    limitations = default_limitations(
        has_evidence=bool(retrieved_sources),
        label=normalized_prediction["label"],
        uncertainty_status=uncertainty_status,
    )
    if rag_warning:
        limitations.append(rag_warning)
    preliminary_report = generate_preliminary_report(
        prediction=normalized_prediction,
        retrieved_sources=retrieved_sources,
        observations=observations_list,
        uncertainty_status=uncertainty_status,
        provider=report_provider,
        prompt=prompt,
        limitations=limitations,
    )
    processing_times["report_seconds"] = elapsed(report_started)
    processing_times["total_seconds"] = elapsed(started_at)

    return {
        "prediction": normalized_prediction["label"],
        "confidence": normalized_prediction["confidence"],
        "probabilities": normalized_prediction["probabilities"],
        "logits": normalized_prediction["logits"],
        "top_3": normalized_prediction["top_3"],
        "uncalibrated_confidence": normalized_prediction["uncalibrated_confidence"],
        "uncalibrated_probabilities": normalized_prediction["uncalibrated_probabilities"],
        "calibration_temperature": normalized_prediction["calibration_temperature"],
        "calibration_applied": normalized_prediction["calibration_applied"],
        "second_class": normalized_prediction["second_class"],
        "second_confidence": normalized_prediction["second_confidence"],
        "top1_top2_margin": normalized_prediction["top1_top2_margin"],
        "uncertainty_status": uncertainty_status,
        "reliability_status": "incierto" if uncertainty_status == "uncertain" else "confiable",
        "retrieved_sources": retrieved_sources,
        "preliminary_report": preliminary_report,
        "limitations": limitations,
        "processing_times": processing_times,
        "retrieval_query": retrieval_query,
        "rag_status": rag_status,
        "rag_warning": rag_warning,
    }


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file as a dictionary."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"La configuracion debe ser un objeto YAML: {path}")
    return loaded


def default_checkpoint_path(config: dict[str, Any]) -> Path:
    """Return the default ResNet18 checkpoint path from the vision config."""
    model_dir = Path(str(get_nested(config, ("output", "model_dir"), "models/vision")))
    return model_dir / "resnet18_baseline_best.pt"


def build_faiss_retriever(
    rag_config: dict[str, Any],
    index_dir: str | Path,
    top_k: int,
) -> "FaissRetriever":
    """Build a FAISS retriever from RAG configuration."""
    from src.rag.retrieval import FaissRetriever

    embedding_model = str(
        get_nested(rag_config, ("rag", "embedding_model"), "sentence-transformers/all-MiniLM-L6-v2")
    )
    normalize_embeddings = bool(get_nested(rag_config, ("rag", "normalize_embeddings"), True))
    return FaissRetriever.from_paths(
        index_dir=index_dir,
        embedding_model=embedding_model,
        top_k=top_k,
        normalize_embeddings=normalize_embeddings,
    )


def build_available_retriever(
    rag_config: dict[str, Any],
    index_dir: str | Path,
    top_k: int,
) -> tuple[Retriever | None, str, str]:
    """Build the best local retriever available without requiring network access."""
    try:
        return build_faiss_retriever(rag_config=rag_config, index_dir=index_dir, top_k=top_k), "faiss", ""
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        metadata_path = Path(index_dir) / "metadata.json"
        if metadata_path.exists():
            from src.rag.retrieval import MetadataKeywordRetriever

            warning = (
                "El recuperador FAISS con embeddings no estuvo disponible; "
                "se uso recuperacion lexical local sobre metadata RAG."
            )
            return MetadataKeywordRetriever.from_path(metadata_path, top_k=top_k), "metadata_fallback", warning
        warning = f"RAG no disponible localmente: {exc.__class__.__name__}."
        return None, "unavailable", warning


def normalize_observations(observations: list[str] | str | None) -> list[str]:
    """Normalize optional observations to a list of non-empty strings."""
    if observations is None:
        return []
    if isinstance(observations, str):
        candidates = [observations]
    else:
        candidates = observations
    return [str(item).strip() for item in candidates if str(item).strip()]


def normalize_prediction(prediction: dict) -> dict:
    """Normalize model prediction keys and numeric values."""
    label = str(prediction.get("label") or prediction.get("prediction") or "unknown")
    confidence = float(prediction.get("confidence", 0.0))
    raw_probabilities = prediction.get("probabilities") or {label: confidence}
    if not isinstance(raw_probabilities, dict):
        raise ValueError("prediction['probabilities'] debe ser un diccionario.")
    probabilities = {str(key): float(value) for key, value in raw_probabilities.items()}
    raw_logits = prediction.get("logits") or {}
    if raw_logits and not isinstance(raw_logits, dict):
        raise ValueError("prediction['logits'] debe ser un diccionario.")
    logits = {str(key): float(value) for key, value in dict(raw_logits).items()}
    raw_top_3 = prediction.get("top_3") or []
    if raw_top_3 and not isinstance(raw_top_3, list):
        raise ValueError("prediction['top_3'] debe ser una lista.")
    top_3 = [dict(item) for item in raw_top_3 if isinstance(item, dict)]
    sorted_probabilities = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    second_class = prediction.get("second_class")
    second_confidence = prediction.get("second_confidence")
    if sorted_probabilities:
        if not top_3:
            top_3 = [
                {"label": item_label, "probability": item_probability}
                for item_label, item_probability in sorted_probabilities[:3]
            ]
        if second_class is None and len(sorted_probabilities) >= 2:
            second_class = sorted_probabilities[1][0]
        if second_confidence is None and len(sorted_probabilities) >= 2:
            second_confidence = sorted_probabilities[1][1]
    top1_top2_margin = prediction.get("top1_top2_margin")
    if top1_top2_margin is None:
        top1_top2_margin = (
            float(sorted_probabilities[0][1]) - float(sorted_probabilities[1][1])
            if len(sorted_probabilities) >= 2
            else confidence
        )
    raw_uncalibrated_probabilities = prediction.get("uncalibrated_probabilities") or probabilities
    if not isinstance(raw_uncalibrated_probabilities, dict):
        raise ValueError("prediction['uncalibrated_probabilities'] debe ser un diccionario.")
    uncalibrated_probabilities = {
        str(key): float(value) for key, value in raw_uncalibrated_probabilities.items()
    }
    uncalibrated_confidence = float(
        prediction.get("uncalibrated_confidence", uncalibrated_probabilities.get(label, confidence))
    )
    calibration_temperature = prediction.get("calibration_temperature")
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "logits": logits,
        "top_3": top_3,
        "uncalibrated_confidence": uncalibrated_confidence,
        "uncalibrated_probabilities": uncalibrated_probabilities,
        "calibration_temperature": (
            float(calibration_temperature) if calibration_temperature is not None else None
        ),
        "calibration_applied": bool(prediction.get("calibration_applied", False)),
        "second_class": str(second_class) if second_class is not None else None,
        "second_confidence": float(second_confidence) if second_confidence is not None else None,
        "top1_top2_margin": float(top1_top2_margin),
    }


def compute_uncertainty_status(
    probabilities: dict[str, float],
    confidence: float,
    confidence_threshold: float,
    margin_threshold: float,
) -> str:
    """Return uncertain when confidence is low or the top-two margin is narrow."""
    sorted_probabilities = sorted(probabilities.values(), reverse=True)
    top_margin = (
        sorted_probabilities[0] - sorted_probabilities[1]
        if len(sorted_probabilities) >= 2
        else confidence
    )
    if confidence < confidence_threshold or top_margin < margin_threshold:
        return "uncertain"
    return "certain"


def call_retriever(
    retriever: Callable[..., list[dict]],
    query: str,
    top_k: int,
) -> list[dict]:
    """Call retrievers that accept either query or query plus top_k."""
    try:
        return retriever(query, top_k=top_k)
    except TypeError:
        return retriever(query)


def get_nested(config: dict[str, Any], keys: tuple[str, ...], default: object) -> object:
    """Read a nested config value with a default."""
    value: object = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def elapsed(started_at: float) -> float:
    """Return rounded elapsed wall-clock seconds."""
    return round(time.perf_counter() - started_at, 6)


def resolve_device(device_name: str | None = None) -> torch.device:
    """Resolve a requested torch device without importing training dependencies."""
    if device_name:
        requested = torch.device(device_name)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
