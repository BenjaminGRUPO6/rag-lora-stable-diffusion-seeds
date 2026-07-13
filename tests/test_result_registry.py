from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.result_registry import (
    BASELINE_EXPERIMENT_ID,
    IMPROVEMENTS_EXPERIMENT_ID,
    ResultRegistry,
)


VISION_RESULTS_DIR = Path("results") / "vision"
BASELINE_COPY_DIR = VISION_RESULTS_DIR / BASELINE_EXPERIMENT_ID
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}


def test_vision_result_structure_exists() -> None:
    """Required Resultados 1 and Resultados 2 directories must exist."""
    expected_dirs = [
        BASELINE_COPY_DIR,
        VISION_RESULTS_DIR / IMPROVEMENTS_EXPERIMENT_ID / "01_metricas_reconciliadas",
        VISION_RESULTS_DIR / IMPROVEMENTS_EXPERIMENT_ID / "02_paridad_inferencia",
        VISION_RESULTS_DIR / IMPROVEMENTS_EXPERIMENT_ID / "03_recorte_y_calidad",
        VISION_RESULTS_DIR / IMPROVEMENTS_EXPERIMENT_ID / "04_analisis_errores",
        VISION_RESULTS_DIR / IMPROVEMENTS_EXPERIMENT_ID / "05_resnet18_v2",
        VISION_RESULTS_DIR / IMPROVEMENTS_EXPERIMENT_ID / "06_calibracion",
        VISION_RESULTS_DIR / IMPROVEMENTS_EXPERIMENT_ID / "07_tta",
        VISION_RESULTS_DIR / IMPROVEMENTS_EXPERIMENT_ID / "08_comparacion_modelos",
        VISION_RESULTS_DIR / IMPROVEMENTS_EXPERIMENT_ID / "09_gradcam_interfaz",
        VISION_RESULTS_DIR / IMPROVEMENTS_EXPERIMENT_ID / "10_lora_generativo",
        VISION_RESULTS_DIR / IMPROVEMENTS_EXPERIMENT_ID / "final",
    ]

    for directory in expected_dirs:
        assert directory.is_dir()


def test_resultados_1_manifest_hashes_match_files() -> None:
    """The baseline copy manifest must keep origin, state and SHA-256 for each file."""
    manifest = json.loads((BASELINE_COPY_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["experiment_id"] == BASELINE_EXPERIMENT_ID
    assert manifest["status"] == "UNRECONCILED"
    assert "discrepancia de metricas" in manifest["metrics_discrepancy_note"]

    manifest_files = {Path(entry["file"]).name: entry for entry in manifest["files"]}
    actual_files = {
        path.name
        for path in BASELINE_COPY_DIR.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }

    assert set(manifest_files) == actual_files
    for file_name, entry in manifest_files.items():
        file_path = BASELINE_COPY_DIR / file_name
        assert entry["status"] == "UNRECONCILED"
        assert Path(entry["origin"]).name == file_name
        assert entry["sha256"] == _sha256(file_path)


def test_resultados_1_contains_no_weights() -> None:
    """The versioned baseline copy must not include checkpoints or model weights."""
    copied_suffixes = {path.suffix.lower() for path in BASELINE_COPY_DIR.rglob("*") if path.is_file()}

    assert copied_suffixes.isdisjoint(WEIGHT_SUFFIXES)


def test_result_registry_protects_resultados_1(tmp_path: Path) -> None:
    """The registry refuses writes to Resultados 1 by default."""
    registry = ResultRegistry(results_root=tmp_path / "vision")

    with pytest.raises(PermissionError):
        registry.artifact_path(BASELINE_EXPERIMENT_ID, "metrics.json")


def test_result_registry_builds_paths_and_refuses_overwrite(tmp_path: Path) -> None:
    """Artifact paths stay inside the experiment and existing files are protected."""
    registry = ResultRegistry(results_root=tmp_path / "vision")
    artifact = registry.artifact_path(
        IMPROVEMENTS_EXPERIMENT_ID,
        Path("01_metricas_reconciliadas") / "metrics.json",
    )
    artifact.write_text("{}", encoding="utf-8")

    assert artifact == (
        tmp_path
        / "vision"
        / IMPROVEMENTS_EXPERIMENT_ID
        / "01_metricas_reconciliadas"
        / "metrics.json"
    )
    with pytest.raises(FileExistsError):
        registry.artifact_path(
            IMPROVEMENTS_EXPERIMENT_ID,
            Path("01_metricas_reconciliadas") / "metrics.json",
        )
    with pytest.raises(ValueError):
        registry.artifact_path(IMPROVEMENTS_EXPERIMENT_ID, Path("..") / "escape.json")


def test_result_registry_registers_experiment_metadata(tmp_path: Path) -> None:
    """Experiment registration records id, date, commit, config and seed."""
    registry = ResultRegistry(results_root=tmp_path / "vision")

    metadata_path = registry.register_experiment(
        "05_resnet18_v2",
        config={"model": {"architecture": "resnet18"}},
        seed=42,
        commit="abc123",
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "05_resnet18_v2"
    assert payload["created_at_utc"].endswith("Z")
    assert payload["commit"] == "abc123"
    assert payload["config"]["model"]["architecture"] == "resnet18"
    assert payload["seed"] == 42


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
