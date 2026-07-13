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
    build_inference_transform,
    load_resnet18_checkpoint,
    predict_image,
)


DEFAULT_VISION_CONFIG = Path("configs/vision_config.yaml")
DEFAULT_RAG_CONFIG = Path("configs/rag.yaml")
DEFAULT_INDEX_DIR = Path("vector_db")

if TYPE_CHECKING:
    from src.rag.retrieval import FaissRetriever


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
        active_checkpoint = checkpoint_path or default_checkpoint_path(vision_config)
        active_model = model
        active_labels = labels
        if active_model is None or active_labels is None:
            active_model, active_labels, _ = load_resnet18_checkpoint(
                checkpoint_path=active_checkpoint,
                device=device,
                config=vision_config,
            )
        active_transform = transform or build_inference_transform(
            image_size=int(get_nested(vision_config, ("data", "image_size"), 224))
        )
        prediction = predict_image(
            model=active_model,
            image_path=image,
            transform=active_transform,
            labels=active_labels,
            device=device,
        )
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
    active_retriever = retriever or build_faiss_retriever(
        rag_config=rag_config,
        index_dir=index_dir,
        top_k=active_top_k,
    )
    retrieved_sources = call_retriever(active_retriever, retrieval_query, active_top_k)
    processing_times["retrieval_seconds"] = elapsed(retrieval_started)

    report_started = time.perf_counter()
    prompt = build_report_prompt(normalized_prediction, retrieved_sources)
    limitations = default_limitations(
        has_evidence=bool(retrieved_sources),
        label=normalized_prediction["label"],
        uncertainty_status=uncertainty_status,
    )
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
        "uncertainty_status": uncertainty_status,
        "retrieved_sources": retrieved_sources,
        "preliminary_report": preliminary_report,
        "limitations": limitations,
        "processing_times": processing_times,
        "retrieval_query": retrieval_query,
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
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
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
