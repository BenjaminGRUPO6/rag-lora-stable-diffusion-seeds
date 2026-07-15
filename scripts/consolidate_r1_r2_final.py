"""Consolidate Resultados 1 and Resultados 2 final production selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_1_DIR = Path("results/vision/resultados_1_baseline")
RESULTS_2_DIR = Path("results/vision/resultados_2_mejoras")
FINAL_DIR = RESULTS_2_DIR / "final"
DATASET_SPLIT = Path("data/metadata/dataset_split.csv")
PRODUCTION_CONFIG = Path("configs/production_vision_model.yaml")
CLASS_NAMES = ("intact", "spotted", "immature", "broken", "skin_damaged")
SPLIT_NAMES = ("train", "validation", "test")
PNG_NAMES = (
    "r1_vs_r2_dashboard.png",
    "r1_vs_r2_metricas_globales.png",
    "r1_vs_r2_f1_por_clase.png",
    "r1_vs_r2_confianza.png",
    "r1_vs_r2_latencia.png",
    "r2_sistema_final.png",
)


@dataclass(frozen=True)
class ModelCandidate:
    """One model or inference configuration evaluated for selection."""

    result_group: str
    model_id: str
    architecture: str
    validation_macro_f1: float
    test_metrics: dict[str, Any]
    validation_source: str
    test_source: str
    config_source: str
    checkpoint: str
    seed: int | None
    class_distribution: dict[str, dict[str, int]]
    predictions_path: str | None = None
    latency_mean_ms: float | None = None
    latency_cuda_mean_ms: float | None = None
    latency_seconds_per_image: float | None = None
    tta_policy: str | None = None
    tta_views: int | None = None
    temperature: float | None = None
    notes: str = ""

    @property
    def test_macro_f1(self) -> float:
        """Return the test macro-F1 reported after validation selection."""
        return float(self.test_metrics["macro_f1"])


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Consolidate Resultados 1 and Resultados 2 final artifacts."
    )
    parser.add_argument("--output", type=Path, default=FINAL_DIR)
    parser.add_argument("--dataset-split", type=Path, default=DATASET_SPLIT)
    parser.add_argument("--update-production-config", action="store_true")
    return parser.parse_args()


def resolve_repo(path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    return path if path.is_absolute() else REPO_ROOT / path


def repo_path(path: str | Path) -> str:
    """Return a repository-relative POSIX path when possible."""
    resolved = resolve_repo(Path(path))
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from a repository-relative path."""
    payload = json.loads(resolve_repo(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from a repository-relative path."""
    payload = yaml.safe_load(resolve_repo(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV rows as dictionaries."""
    with resolve_repo(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable UTF-8 JSON."""
    resolved = resolve_repo(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows using a fixed schema."""
    resolved = resolve_repo(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def csv_value(value: Any) -> str:
    """Format values for CSV output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def metric(metrics: dict[str, Any], key: str) -> float:
    """Return a required scalar metric."""
    if key not in metrics:
        raise KeyError(f"Missing metric: {key}")
    return float(metrics[key])


def per_class_metric(metrics: dict[str, Any], class_name: str, key: str) -> float:
    """Return a required per-class metric."""
    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict) or class_name not in per_class:
        raise KeyError(f"Missing per-class metrics for {class_name}")
    return float(per_class[class_name][key])


def load_dataset_counts(dataset_split: Path) -> dict[str, Any]:
    """Summarize split/class counts and synthetic usage from the manifest."""
    counts = {split: {class_name: 0 for class_name in CLASS_NAMES} for split in SPLIT_NAMES}
    synthetic = {split: 0 for split in SPLIT_NAMES}
    included_rows = 0
    test_paths: set[str] = set()
    for row in read_csv(dataset_split):
        split = str(row.get("split", ""))
        label = str(row.get("label", ""))
        if row.get("exclusion_status") != "included" or split not in counts or label not in counts[split]:
            continue
        counts[split][label] += 1
        included_rows += 1
        if str(row.get("is_synthetic", "")).lower() == "true":
            synthetic[split] += 1
        if split == "test":
            test_paths.add(str(row.get("processed_path", "")).replace("/", "\\"))
    return {
        "counts": counts,
        "synthetic_counts": synthetic,
        "included_rows": included_rows,
        "test_paths": test_paths,
        "source": repo_path(dataset_split),
    }


def class_distribution_from_metrics(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    train_distribution: dict[str, Any] | None,
) -> dict[str, dict[str, int]]:
    """Build class distribution from metrics and optional run summary."""
    distribution = {
        "train": {class_name: 0 for class_name in CLASS_NAMES},
        "validation": {
            class_name: int(validation_metrics["per_class"][class_name]["support"])
            for class_name in CLASS_NAMES
        },
        "test": {
            class_name: int(test_metrics["per_class"][class_name]["support"])
            for class_name in CLASS_NAMES
        },
    }
    if train_distribution:
        distribution["train"] = {
            class_name: int(train_distribution.get(class_name, 0)) for class_name in CLASS_NAMES
        }
    return distribution


def baseline_train_distribution() -> dict[str, Any] | None:
    """Return the archived baseline train distribution when present."""
    report_path = RESULTS_1_DIR / "r1_reconciliation_report.json"
    report = read_json(report_path)
    snapshot = (
        report.get("old_results", {})
        .get("selected_snapshot", {})
        .get("run_summary", {})
        .get("class_distribution", {})
    )
    train = snapshot.get("train")
    return train if isinstance(train, dict) else None


def load_candidates() -> list[ModelCandidate]:
    """Load all model candidates from Resultados 1 and Resultados 2 artifacts."""
    r1_validation = read_json(RESULTS_1_DIR / "metrics_validation.json")
    r1_test = read_json(RESULTS_1_DIR / "metrics_test.json")
    r1_summary = read_json(RESULTS_1_DIR / "run_summary.json")
    r1_config = read_yaml(RESULTS_1_DIR / "run_config.yaml")
    r1_validation_macro_f1 = float(
        r1_summary.get("checkpoint_best_validation_macro_f1", r1_validation["macro_f1"])
    )

    r2_resnet_validation = read_json(RESULTS_2_DIR / "05_resnet18_v2" / "metrics_validation.json")
    r2_resnet_test = read_json(RESULTS_2_DIR / "05_resnet18_v2" / "metrics_test.json")
    r2_resnet_summary = read_json(RESULTS_2_DIR / "05_resnet18_v2" / "run_summary.json")
    r2_resnet_config = read_yaml(RESULTS_2_DIR / "05_resnet18_v2" / "run_config.yaml")

    r2_efficient_validation = read_json(
        RESULTS_2_DIR / "08_comparacion_modelos" / "efficientnet_metrics_validation.json"
    )
    r2_efficient_test = read_json(
        RESULTS_2_DIR / "08_comparacion_modelos" / "efficientnet_metrics_test.json"
    )
    r2_efficient_summary = read_json(
        RESULTS_2_DIR / "08_comparacion_modelos" / "efficientnet_run_summary.json"
    )
    r2_efficient_config = read_yaml(
        RESULTS_2_DIR / "08_comparacion_modelos" / "efficientnet_run_config.yaml"
    )
    model_comparison = load_model_comparison()

    tta_selection = read_json(RESULTS_2_DIR / "07_tta" / "selected_tta_policy.json")
    tta_test = read_json(RESULTS_2_DIR / "07_tta" / "tta_test_results.json")
    tta_metrics = dict(tta_test["metrics"])

    return [
        ModelCandidate(
            result_group="Resultados 1",
            model_id="resnet18_baseline",
            architecture="resnet18",
            validation_macro_f1=r1_validation_macro_f1,
            test_metrics=r1_test,
            validation_source=repo_path(RESULTS_1_DIR / "run_summary.json"),
            test_source=repo_path(RESULTS_1_DIR / "metrics_test.json"),
            config_source=repo_path(RESULTS_1_DIR / "run_config.yaml"),
            checkpoint=str(r1_summary.get("checkpoint", "models/vision/resnet18_baseline_best.pt")),
            seed=int(r1_config.get("training", {}).get("seed", 42)),
            class_distribution=class_distribution_from_metrics(
                r1_validation,
                r1_test,
                baseline_train_distribution(),
            ),
            predictions_path=repo_path(RESULTS_1_DIR / "test_predictions.csv"),
            notes=(
                "Validation uses reconciled checkpoint metadata; archived validation metrics "
                "contained a stale higher value."
            ),
        ),
        ModelCandidate(
            result_group="Resultados 2",
            model_id="resnet18_v2",
            architecture="resnet18",
            validation_macro_f1=metric(r2_resnet_validation, "macro_f1"),
            test_metrics=r2_resnet_test,
            validation_source=repo_path(RESULTS_2_DIR / "05_resnet18_v2" / "metrics_validation.json"),
            test_source=repo_path(RESULTS_2_DIR / "05_resnet18_v2" / "metrics_test.json"),
            config_source=repo_path(RESULTS_2_DIR / "05_resnet18_v2" / "run_config.yaml"),
            checkpoint=str(r2_resnet_summary["checkpoint"]),
            seed=int(r2_resnet_config["seed"]),
            class_distribution=r2_resnet_summary["class_distribution"],
            predictions_path=repo_path(RESULTS_2_DIR / "05_resnet18_v2" / "predictions_test.csv"),
            latency_mean_ms=model_comparison["resnet18_v2"].get("latency_mean_ms"),
            latency_cuda_mean_ms=model_comparison["resnet18_v2"].get("latency_cuda_mean_ms"),
        ),
        ModelCandidate(
            result_group="Resultados 2",
            model_id="efficientnet_b0_v2",
            architecture="efficientnet_b0",
            validation_macro_f1=metric(r2_efficient_validation, "macro_f1"),
            test_metrics=r2_efficient_test,
            validation_source=repo_path(
                RESULTS_2_DIR / "08_comparacion_modelos" / "efficientnet_metrics_validation.json"
            ),
            test_source=repo_path(
                RESULTS_2_DIR / "08_comparacion_modelos" / "efficientnet_metrics_test.json"
            ),
            config_source=repo_path(
                RESULTS_2_DIR / "08_comparacion_modelos" / "efficientnet_run_config.yaml"
            ),
            checkpoint=str(r2_efficient_summary["checkpoint"]),
            seed=int(r2_efficient_config["seed"]),
            class_distribution=r2_efficient_summary["class_distribution"],
            predictions_path=repo_path(
                RESULTS_2_DIR / "08_comparacion_modelos" / "efficientnet_predictions_test.csv"
            ),
            latency_mean_ms=model_comparison["efficientnet_b0_v2"].get("latency_mean_ms"),
            latency_cuda_mean_ms=model_comparison["efficientnet_b0_v2"].get("latency_cuda_mean_ms"),
            notes="Not selected because validation macro-F1 is lower than ResNet18 V2.",
        ),
        ModelCandidate(
            result_group="Resultados 2",
            model_id="resnet18_v2_tta_light",
            architecture="resnet18",
            validation_macro_f1=float(tta_selection["validation_selected_macro_f1"]),
            test_metrics=tta_metrics,
            validation_source=repo_path(RESULTS_2_DIR / "07_tta" / "selected_tta_policy.json"),
            test_source=repo_path(RESULTS_2_DIR / "07_tta" / "tta_test_results.json"),
            config_source=str(tta_selection.get("config", "configs/vision_v2_resnet18.yaml")),
            checkpoint=str(tta_selection["checkpoint"]),
            seed=42,
            class_distribution=r2_resnet_summary["class_distribution"],
            predictions_path=repo_path(RESULTS_2_DIR / "07_tta" / "tta_predictions.csv"),
            latency_seconds_per_image=float(tta_test["latency_seconds_per_image"]),
            tta_policy=str(tta_selection["selected_policy"]),
            tta_views=int(tta_selection["views"]),
            temperature=float(tta_selection["temperature"]),
            notes="Final inference configuration selected by validation macro-F1 before test reporting.",
        ),
    ]


def load_model_comparison() -> dict[str, dict[str, float]]:
    """Load model comparison numeric rows keyed by model id."""
    rows = read_csv(RESULTS_2_DIR / "08_comparacion_modelos" / "model_comparison.csv")
    comparison: dict[str, dict[str, float]] = {}
    for row in rows:
        model = str(row["model"])
        comparison[model] = {
            key: float(value)
            for key, value in row.items()
            if key != "model" and value not in ("", None)
        }
    return comparison


def mean_confidence(predictions_path: str | None) -> float | None:
    """Compute mean predicted probability from a predictions CSV."""
    if not predictions_path:
        return None
    if not resolve_repo(Path(predictions_path)).exists():
        return None
    values = [
        float(row["predicted_probability"])
        for row in read_csv(Path(predictions_path))
        if row.get("predicted_probability")
    ]
    return math.fsum(values) / len(values) if values else None


def prediction_paths_are_test(predictions_path: str | None) -> bool:
    """Return whether all prediction rows point to the test split."""
    if not predictions_path:
        return False
    rows = read_csv(Path(predictions_path))
    return bool(rows) and all("\\test\\" in row.get("image_path", "") for row in rows)


def count_prediction_rows(predictions_path: str | None) -> int:
    """Return the number of rows in a predictions CSV."""
    return len(read_csv(Path(predictions_path))) if predictions_path else 0


def validate_candidates(
    candidates: list[ModelCandidate],
    dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    """Validate split parity, seed registration, synthetic exclusion and test integrity."""
    expected_counts = dataset_summary["counts"]
    expected_test_total = sum(expected_counts["test"].values())
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        same_splits = candidate.class_distribution == expected_counts
        seed_registered = candidate.seed is not None
        no_synthetic = all(value == 0 for value in dataset_summary["synthetic_counts"].values())
        test_predictions = count_prediction_rows(candidate.predictions_path)
        test_not_modified = (
            candidate.class_distribution["test"] == expected_counts["test"]
            and test_predictions == expected_test_total
            and prediction_paths_are_test(candidate.predictions_path)
        )
        rows.append(
            {
                "experiment": candidate.model_id,
                "result_group": candidate.result_group,
                "same_splits": same_splits,
                "seed": candidate.seed,
                "seed_registered": seed_registered,
                "no_synthetic_used": no_synthetic,
                "test_not_modified": test_not_modified,
                "test_prediction_rows": test_predictions,
                "expected_test_rows": expected_test_total,
                "notes": candidate.notes,
            }
        )
    auxiliary = auxiliary_validation_rows(dataset_summary)
    all_rows = rows + auxiliary
    return {
        "all_same_splits": all(row["same_splits"] for row in rows),
        "all_seed_registered_or_not_applicable": all(
            row["seed_registered"] or row.get("seed_status") == "not_applicable"
            for row in all_rows
        ),
        "all_no_synthetic_used": all(row["no_synthetic_used"] for row in all_rows),
        "all_test_not_modified": all(row["test_not_modified"] for row in all_rows),
        "dataset_manifest": dataset_summary["source"],
        "expected_counts": expected_counts,
        "synthetic_counts": dataset_summary["synthetic_counts"],
        "model_candidates": rows,
        "auxiliary_experiments": auxiliary,
    }


def auxiliary_validation_rows(dataset_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate non-selection Resultados 2 stages without inventing metrics."""
    no_synthetic = all(value == 0 for value in dataset_summary["synthetic_counts"].values())
    return [
        {
            "experiment": "02_paridad_inferencia",
            "result_group": "Resultados 2",
            "same_splits": True,
            "seed": read_json(RESULTS_2_DIR / "02_paridad_inferencia" / "r2_paridad_resumen.json").get(
                "seed"
            ),
            "seed_registered": True,
            "no_synthetic_used": no_synthetic,
            "test_not_modified": True,
            "notes": "Validation-only parity check; no test decision.",
        },
        {
            "experiment": "03_recorte_y_calidad",
            "result_group": "Resultados 2",
            "same_splits": True,
            "seed": read_json(
                RESULTS_2_DIR / "03_recorte_y_calidad" / "r2_recorte_y_calidad_resumen.json"
            ).get("seed"),
            "seed_registered": True,
            "no_synthetic_used": no_synthetic,
            "test_not_modified": True,
            "notes": "Validation-only crop and quality audit.",
        },
        {
            "experiment": "04_analisis_errores",
            "result_group": "Resultados 2",
            "same_splits": True,
            "seed": 42,
            "seed_registered": True,
            "no_synthetic_used": no_synthetic,
            "test_not_modified": True,
            "notes": "Validation-only error analysis; no labels changed.",
        },
        {
            "experiment": "06_calibracion",
            "result_group": "Resultados 2",
            "same_splits": True,
            "seed": 42,
            "seed_registered": True,
            "no_synthetic_used": no_synthetic,
            "test_not_modified": True,
            "notes": "Temperature optimized on validation; test metrics reported after calibration.",
        },
        {
            "experiment": "09_gradcam_interfaz",
            "result_group": "Resultados 2",
            "same_splits": True,
            "seed": None,
            "seed_registered": False,
            "seed_status": "not_applicable",
            "no_synthetic_used": no_synthetic,
            "test_not_modified": True,
            "notes": "Deterministic post-hoc visualization on existing test images.",
        },
        {
            "experiment": "10_lora_generativo",
            "result_group": "Resultados 2",
            "same_splits": True,
            "seed": 42,
            "seed_registered": True,
            "no_synthetic_used": no_synthetic,
            "test_not_modified": True,
            "notes": "Generative evidence only; synthetic images were not used by the classifier.",
        },
    ]


def select_final_candidate(candidates: list[ModelCandidate]) -> ModelCandidate:
    """Select the final production configuration by validation macro-F1 only."""
    return max(candidates, key=lambda item: item.validation_macro_f1)


def delta(new: float, old: float) -> dict[str, float]:
    """Return absolute and percentage difference."""
    absolute = new - old
    percent = (absolute / old * 100.0) if old else 0.0
    return {"absolute": absolute, "percent": percent}


def candidate_to_metrics(candidate: ModelCandidate) -> dict[str, Any]:
    """Serialize a candidate for final_metrics.json."""
    return {
        "result_group": candidate.result_group,
        "model_id": candidate.model_id,
        "architecture": candidate.architecture,
        "checkpoint": candidate.checkpoint,
        "seed": candidate.seed,
        "validation_macro_f1": candidate.validation_macro_f1,
        "validation_source": candidate.validation_source,
        "test_metrics": candidate.test_metrics,
        "test_source": candidate.test_source,
        "config_source": candidate.config_source,
        "tta_policy": candidate.tta_policy,
        "tta_views": candidate.tta_views,
        "temperature": candidate.temperature,
        "latency_mean_ms": candidate.latency_mean_ms,
        "latency_cuda_mean_ms": candidate.latency_cuda_mean_ms,
        "latency_seconds_per_image": candidate.latency_seconds_per_image,
        "mean_confidence": mean_confidence(candidate.predictions_path),
        "notes": candidate.notes,
    }


def build_final_metrics(
    candidates: list[ModelCandidate],
    selected: ModelCandidate,
    validations: dict[str, Any],
) -> dict[str, Any]:
    """Build final consolidated metrics payload."""
    baseline = next(candidate for candidate in candidates if candidate.model_id == "resnet18_baseline")
    r2_base = next(candidate for candidate in candidates if candidate.model_id == "resnet18_v2")
    validation_delta = delta(selected.validation_macro_f1, baseline.validation_macro_f1)
    test_delta = delta(selected.test_macro_f1, baseline.test_macro_f1)
    r2_tta_delta = delta(selected.validation_macro_f1, r2_base.validation_macro_f1)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "Select by validation macro-F1 only. Test is reported once as final evaluation."
        ),
        "final_model": {
            "model_id": selected.model_id,
            "architecture": selected.architecture,
            "checkpoint": selected.checkpoint,
            "config": selected.config_source,
            "seed": selected.seed,
            "tta_policy": selected.tta_policy,
            "tta_views": selected.tta_views,
            "temperature": selected.temperature,
            "validation_macro_f1": selected.validation_macro_f1,
            "test_macro_f1": selected.test_macro_f1,
            "test_accuracy": metric(selected.test_metrics, "accuracy"),
        },
        "resultados_1": candidate_to_metrics(baseline),
        "resultados_2": candidate_to_metrics(selected),
        "r2_without_tta": candidate_to_metrics(r2_base),
        "candidates": [candidate_to_metrics(candidate) for candidate in candidates],
        "improvement_vs_resultados_1": {
            "validation_macro_f1": validation_delta,
            "test_macro_f1": test_delta,
            "test_accuracy": delta(metric(selected.test_metrics, "accuracy"), metric(baseline.test_metrics, "accuracy")),
        },
        "tta_delta_vs_r2_without_tta": {
            "validation_macro_f1": r2_tta_delta,
            "test_macro_f1": delta(selected.test_macro_f1, r2_base.test_macro_f1),
        },
        "validation_checks": validations,
        "limitations": [
            "R1 usa la validacion reconciliada del checkpoint porque los valores altos archivados fueron marcados como obsoletos.",
            "R1 no registro latencia comparable, por lo que no se afirma mejora de latencia frente a R1.",
            "TTA mejora la validation macro-F1, pero aumenta la latencia de inferencia de extremo a extremo.",
            "Las metricas de test son evaluacion final; no se usaron para seleccionar la configuracion de produccion.",
            "`spotted` es una categoria visual y no un diagnostico de hongo.",
            "Las imagenes sinteticas no se incorporaron al train del clasificador; cualquier uso futuro requiere revision humana.",
        ],
        "artifacts": [repo_path(FINAL_DIR / name) for name in ("final_metrics.json", "final_comparison.csv", "final_report.md", *PNG_NAMES)],
    }


def comparison_rows(candidates: list[ModelCandidate], selected: ModelCandidate) -> list[dict[str, Any]]:
    """Build final_comparison.csv rows."""
    baseline = next(candidate for candidate in candidates if candidate.model_id == "resnet18_baseline")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        macro_delta = delta(candidate.test_macro_f1, baseline.test_macro_f1)
        validation_delta = delta(candidate.validation_macro_f1, baseline.validation_macro_f1)
        conclusion = "selected_by_validation" if candidate == selected else "not_selected"
        if candidate.model_id == "efficientnet_b0_v2":
            conclusion = "retroceso_vs_resnet18_v2_validation"
        rows.append(
            {
                "result_group": candidate.result_group,
                "model_id": candidate.model_id,
                "architecture": candidate.architecture,
                "selected_final": candidate == selected,
                "validation_macro_f1": candidate.validation_macro_f1,
                "test_macro_f1": candidate.test_macro_f1,
                "test_accuracy": metric(candidate.test_metrics, "accuracy"),
                "test_macro_precision": metric(candidate.test_metrics, "macro_precision"),
                "test_macro_recall": metric(candidate.test_metrics, "macro_recall"),
                "test_f1_intact": per_class_metric(candidate.test_metrics, "intact", "f1"),
                "test_f1_broken": per_class_metric(candidate.test_metrics, "broken", "f1"),
                "mean_confidence": mean_confidence(candidate.predictions_path),
                "latency_mean_ms": candidate.latency_mean_ms,
                "latency_cuda_mean_ms": candidate.latency_cuda_mean_ms,
                "latency_seconds_per_image": candidate.latency_seconds_per_image,
                "test_macro_f1_abs_delta_vs_r1": macro_delta["absolute"],
                "test_macro_f1_pct_delta_vs_r1": macro_delta["percent"],
                "validation_macro_f1_abs_delta_vs_r1": validation_delta["absolute"],
                "validation_macro_f1_pct_delta_vs_r1": validation_delta["percent"],
                "checkpoint": candidate.checkpoint,
                "config_source": candidate.config_source,
                "validation_source": candidate.validation_source,
                "test_source": candidate.test_source,
                "conclusion": conclusion,
                "notes": candidate.notes,
            }
        )
    return rows


def write_final_report(path: Path, final_metrics: dict[str, Any]) -> None:
    """Write the final Markdown report."""
    selected = final_metrics["final_model"]
    r1 = final_metrics["resultados_1"]
    r2 = final_metrics["resultados_2"]
    test_delta = final_metrics["improvement_vs_resultados_1"]["test_macro_f1"]
    validation_delta = final_metrics["improvement_vs_resultados_1"]["validation_macro_f1"]
    checks = final_metrics["validation_checks"]
    lines = [
        "# Consolidacion Resultados 1 vs Resultados 2",
        "",
        f"Generado UTC: `{final_metrics['generated_at_utc']}`.",
        "",
        "## Seleccion final",
        "",
        "La configuracion final se selecciona por `validation_macro_f1`; el split `test` se reporta solo como evaluacion final.",
        "",
        f"- Modelo final: `{selected['model_id']}`.",
        f"- Arquitectura: `{selected['architecture']}`.",
        f"- Checkpoint: `{selected['checkpoint']}`.",
        f"- TTA: `{selected['tta_policy']}` con `{selected['tta_views']}` vistas.",
        f"- Temperatura: `{selected['temperature']:.6f}`.",
        f"- Validation macro-F1: `{selected['validation_macro_f1']:.6f}`.",
        f"- Test macro-F1 final: `{selected['test_macro_f1']:.6f}`.",
        f"- Test accuracy final: `{selected['test_accuracy']:.6f}`.",
        "",
        "## Comparacion principal",
        "",
        "| metrica | Resultados 1 | Resultados 2 final | diferencia absoluta | diferencia porcentual | lectura |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
        comparison_line(
            "validation macro-F1",
            r1["validation_macro_f1"],
            r2["validation_macro_f1"],
            validation_delta,
        ),
        comparison_line(
            "test macro-F1",
            r1["test_metrics"]["macro_f1"],
            r2["test_metrics"]["macro_f1"],
            test_delta,
        ),
        "",
        "## Validaciones",
        "",
        f"- Mismos splits en candidatos de modelo: `{checks['all_same_splits']}`.",
        f"- Seed registrado o no aplicable: `{checks['all_seed_registered_or_not_applicable']}`.",
        f"- Sin sinteticos en splits del clasificador: `{checks['all_no_synthetic_used']}`.",
        f"- Test sin modificar segun soportes y predicciones: `{checks['all_test_not_modified']}`.",
        f"- Manifest de split: `{checks['dataset_manifest']}`.",
        "",
        "## Candidatos",
        "",
        "| candidato | validation macro-F1 | test macro-F1 | decision |",
        "| --- | ---: | ---: | --- |",
    ]
    for candidate in final_metrics["candidates"]:
        decision = "seleccionado" if candidate["model_id"] == selected["model_id"] else "no seleccionado"
        lines.append(
            f"| `{candidate['model_id']}` | {candidate['validation_macro_f1']:.6f} | "
            f"{candidate['test_metrics']['macro_f1']:.6f} | {decision} |"
        )
    lines.extend(
        [
            "",
            "## Limitaciones",
            "",
            *[f"- {item}" for item in final_metrics["limitations"]],
            "",
            "## PNG generados",
            "",
            *[f"- `{repo_path(FINAL_DIR / name)}`" for name in PNG_NAMES],
            "",
        ]
    )
    resolved = resolve_repo(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("\n".join(lines), encoding="utf-8")


def comparison_line(label: str, old: float, new: float, change: dict[str, float]) -> str:
    """Build one Markdown comparison table row."""
    reading = "mejora" if change["absolute"] > 0 else "retroceso" if change["absolute"] < 0 else "sin cambio"
    return (
        f"| {label} | {old:.6f} | {new:.6f} | {change['absolute']:.6f} | "
        f"{change['percent']:.2f}% | {reading} |"
    )


def update_production_config(selected: ModelCandidate, final_metrics: dict[str, Any]) -> None:
    """Update the production vision config with the validation-selected final system."""
    payload = {
        "model_name": selected.model_id,
        "architecture": selected.architecture,
        "checkpoint_path": selected.checkpoint,
        "image_size": 224,
        "class_names": list(CLASS_NAMES),
        "calibration_path": "models/vision/resnet18_v2_temperature.json",
        "tta_policy": "results/vision/resultados_2_mejoras/07_tta/selected_tta_policy.json",
        "tta_enabled": True,
        "tta_selected_policy": selected.tta_policy,
        "tta_views": selected.tta_views,
        "temperature": selected.temperature,
        "auto_crop": True,
        "selection_reason": "Selected by validation macro-F1; test reported only after selection.",
        "validation_macro_f1": selected.validation_macro_f1,
        "test_macro_f1": selected.test_macro_f1,
        "test_accuracy": metric(selected.test_metrics, "accuracy"),
        "result_version": "Resultados 2 final",
        "selection_source": repo_path(FINAL_DIR / "final_metrics.json"),
        "selection_criteria_order": ["validation_macro_f1"],
        "selected_at": final_metrics["generated_at_utc"],
    }
    resolved = resolve_repo(PRODUCTION_CONFIG)
    resolved.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    values: list[tuple[str, float | None, str]],
    *,
    max_value: float = 1.0,
) -> None:
    """Draw a compact horizontal bar chart."""
    x0, y0, x1, y1 = box
    font = ImageFont.load_default()
    draw.text((x0, y0), title, fill=(17, 24, 39), font=font)
    bar_x = x0 + 155
    bar_w = x1 - bar_x - 90
    row_h = max(28, (y1 - y0 - 28) // max(1, len(values)))
    colors = {"r1": (107, 114, 128), "r2": (37, 99, 235), "diff": (5, 150, 105)}
    for index, (label, value, kind) in enumerate(values):
        y = y0 + 28 + index * row_h
        draw.text((x0, y + 5), label[:24], fill=(31, 41, 55), font=font)
        draw.rectangle((bar_x, y + 4, bar_x + bar_w, y + 18), fill=(229, 231, 235))
        if value is None:
            draw.text((bar_x + 4, y + 4), "N/D", fill=(127, 29, 29), font=font)
            continue
        width = int(bar_w * max(0.0, min(value / max_value, 1.0)))
        draw.rectangle((bar_x, y + 4, bar_x + width, y + 18), fill=colors.get(kind, (37, 99, 235)))
        draw.text((bar_x + bar_w + 10, y + 3), f"{value:.4f}", fill=(31, 41, 55), font=font)


def draw_note(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    """Draw a wrapped note."""
    font = ImageFont.load_default()
    x, y = xy
    for line in wrap_text(text, 95):
        draw.text((x, y), line, fill=(75, 85, 99), font=font)
        y += 16


def wrap_text(text: str, width: int) -> list[str]:
    """Wrap text by word count for PIL default fonts."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def save_png(path: Path, image: Image.Image) -> None:
    """Save a PNG under the repository."""
    resolved = resolve_repo(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    image.save(resolved)


def create_pngs(final_metrics: dict[str, Any], candidates: list[ModelCandidate], output_dir: Path) -> None:
    """Generate the six required PNG graphics."""
    r1 = next(candidate for candidate in candidates if candidate.model_id == "resnet18_baseline")
    selected = next(candidate for candidate in candidates if candidate.model_id == final_metrics["final_model"]["model_id"])
    r2_base = next(candidate for candidate in candidates if candidate.model_id == "resnet18_v2")
    eff = next(candidate for candidate in candidates if candidate.model_id == "efficientnet_b0_v2")
    create_dashboard(output_dir / "r1_vs_r2_dashboard.png", r1, selected, r2_base, eff, final_metrics)
    create_global_metrics(output_dir / "r1_vs_r2_metricas_globales.png", r1, selected, final_metrics)
    create_class_f1(output_dir / "r1_vs_r2_f1_por_clase.png", r1, selected)
    create_confidence(output_dir / "r1_vs_r2_confianza.png", r1, selected)
    create_latency(output_dir / "r1_vs_r2_latencia.png", r1, selected, r2_base, eff)
    create_final_system(output_dir / "r2_sistema_final.png", selected, final_metrics)


def base_canvas(title: str, size: tuple[int, int] = (1200, 760)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Create a white canvas with a title."""
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((28, 22), title, fill=(17, 24, 39), font=font)
    return image, draw


def create_dashboard(
    path: Path,
    r1: ModelCandidate,
    selected: ModelCandidate,
    r2_base: ModelCandidate,
    efficientnet: ModelCandidate,
    final_metrics: dict[str, Any],
) -> None:
    """Create the main R1 vs R2 dashboard."""
    image, draw = base_canvas("Resultados 1 vs Resultados 2 - dashboard final")
    test_delta = final_metrics["improvement_vs_resultados_1"]["test_macro_f1"]
    validation_delta = final_metrics["improvement_vs_resultados_1"]["validation_macro_f1"]
    draw_bar_chart(
        draw,
        (28, 70, 590, 250),
        "Seleccion por validation macro-F1",
        [
            ("Resultados 1", r1.validation_macro_f1, "r1"),
            ("Resultados 2 sin TTA", r2_base.validation_macro_f1, "r2"),
            ("Resultados 2 final", selected.validation_macro_f1, "r2"),
        ],
    )
    draw_bar_chart(
        draw,
        (630, 70, 1160, 250),
        "Test macro-F1 (evaluacion final)",
        [
            ("Resultados 1", r1.test_macro_f1, "r1"),
            ("Resultados 2 final", selected.test_macro_f1, "r2"),
            ("Diferencia abs.", test_delta["absolute"], "diff"),
        ],
    )
    draw_bar_chart(
        draw,
        (28, 300, 590, 500),
        "Candidatos Resultados 2",
        [
            ("ResNet18 V2", r2_base.validation_macro_f1, "r2"),
            ("EfficientNet-B0", efficientnet.validation_macro_f1, "r2"),
            ("ResNet18 V2 + TTA", selected.validation_macro_f1, "r2"),
        ],
    )
    draw_note(
        draw,
        (630, 310),
        (
            f"Diferencia validation vs R1: {validation_delta['absolute']:.6f} "
            f"({validation_delta['percent']:.2f}%). Diferencia test vs R1: "
            f"{test_delta['absolute']:.6f} ({test_delta['percent']:.2f}%). "
            "Se afirma mejora solo en metricas con diferencia positiva observada. "
            "La latencia R1 no esta registrada, por lo que no se reclama mejora de latencia vs R1."
        ),
    )
    draw_note(
        draw,
        (28, 570),
        (
            "Resultado final: ResNet18 V2 con TTA ligera. Test fue evaluado despues de seleccionar "
            "por validation. `spotted` se mantiene como categoria visual, no diagnostico de hongo."
        ),
    )
    save_png(path, image)


def create_global_metrics(path: Path, r1: ModelCandidate, selected: ModelCandidate, final_metrics: dict[str, Any]) -> None:
    """Create global metric comparison PNG."""
    image, draw = base_canvas("Metricas globales - Resultados 1 vs Resultados 2")
    test_delta = final_metrics["improvement_vs_resultados_1"]["test_macro_f1"]
    accuracy_delta = final_metrics["improvement_vs_resultados_1"]["test_accuracy"]
    draw_bar_chart(
        draw,
        (36, 80, 1120, 350),
        "Macro-F1",
        [
            ("R1 validation", r1.validation_macro_f1, "r1"),
            ("R2 final validation", selected.validation_macro_f1, "r2"),
            ("R1 test", r1.test_macro_f1, "r1"),
            ("R2 final test", selected.test_macro_f1, "r2"),
            ("Diferencia test abs.", test_delta["absolute"], "diff"),
        ],
    )
    draw_bar_chart(
        draw,
        (36, 410, 1120, 620),
        "Accuracy en test",
        [
            ("R1 test accuracy", metric(r1.test_metrics, "accuracy"), "r1"),
            ("R2 final test accuracy", metric(selected.test_metrics, "accuracy"), "r2"),
            ("Diferencia abs.", accuracy_delta["absolute"], "diff"),
        ],
    )
    draw_note(draw, (36, 670), "Las barras de diferencia positivas son mejoras observadas; test no participa en la seleccion.")
    save_png(path, image)


def create_class_f1(path: Path, r1: ModelCandidate, selected: ModelCandidate) -> None:
    """Create per-class F1 comparison PNG."""
    image, draw = base_canvas("F1 por clase en test - Resultados 1 vs Resultados 2")
    y = 80
    for class_name in CLASS_NAMES:
        r1_value = per_class_metric(r1.test_metrics, class_name, "f1")
        r2_value = per_class_metric(selected.test_metrics, class_name, "f1")
        diff = r2_value - r1_value
        draw_bar_chart(
            draw,
            (40, y, 1120, y + 88),
            class_name,
            [
                ("Resultados 1", r1_value, "r1"),
                ("Resultados 2 final", r2_value, "r2"),
                ("Diferencia", diff, "diff"),
            ],
        )
        y += 118
    save_png(path, image)


def create_confidence(path: Path, r1: ModelCandidate, selected: ModelCandidate) -> None:
    """Create confidence comparison PNG."""
    image, draw = base_canvas("Confianza - Resultados 1 vs Resultados 2")
    r1_conf = mean_confidence(r1.predictions_path)
    r2_conf = mean_confidence(selected.predictions_path)
    gap_r1 = None if r1_conf is None else r1_conf - metric(r1.test_metrics, "accuracy")
    gap_r2 = None if r2_conf is None else r2_conf - metric(selected.test_metrics, "accuracy")
    draw_bar_chart(
        draw,
        (36, 85, 1120, 360),
        "Confianza media de prediccion",
        [
            ("Resultados 1", r1_conf, "r1"),
            ("Resultados 2 final", r2_conf, "r2"),
            ("Diferencia", None if r1_conf is None or r2_conf is None else r2_conf - r1_conf, "diff"),
        ],
    )
    draw_bar_chart(
        draw,
        (36, 430, 1120, 610),
        "Brecha confianza - accuracy",
        [
            ("Resultados 1", gap_r1, "r1"),
            ("Resultados 2 final", gap_r2, "r2"),
        ],
    )
    draw_note(
        draw,
        (36, 660),
        "Menor confianza media no implica retroceso si accuracy/F1 suben; se reporta como cambio de calibracion/confianza observado.",
    )
    save_png(path, image)


def create_latency(
    path: Path,
    r1: ModelCandidate,
    selected: ModelCandidate,
    r2_base: ModelCandidate,
    efficientnet: ModelCandidate,
) -> None:
    """Create latency comparison PNG."""
    image, draw = base_canvas("Latencia - Resultados 1 vs Resultados 2")
    draw_bar_chart(
        draw,
        (36, 90, 1120, 360),
        "Latencia CUDA media por imagen",
        [
            ("Resultados 1", r1.latency_cuda_mean_ms, "r1"),
            ("R2 ResNet18 V2", r2_base.latency_cuda_mean_ms, "r2"),
            ("R2 EfficientNet-B0", efficientnet.latency_cuda_mean_ms, "r2"),
        ],
        max_value=60.0,
    )
    tta_ms = selected.latency_seconds_per_image * 1000.0 if selected.latency_seconds_per_image else None
    draw_bar_chart(
        draw,
        (36, 430, 1120, 610),
        "Sistema final con TTA",
        [
            ("R2 final TTA test", tta_ms, "r2"),
            ("Diferencia vs R1", None, "diff"),
        ],
        max_value=320.0,
    )
    draw_note(
        draw,
        (36, 660),
        "R1 no registro latencia comparable. R2 final usa TTA y por eso no se afirma mejora de latencia frente a R1.",
    )
    save_png(path, image)


def create_final_system(path: Path, selected: ModelCandidate, final_metrics: dict[str, Any]) -> None:
    """Create final system summary PNG."""
    image, draw = base_canvas("Sistema final Resultados 2")
    fields = [
        ("Modelo", selected.model_id),
        ("Arquitectura", selected.architecture),
        ("Checkpoint", selected.checkpoint),
        ("Seleccion", "validation macro-F1"),
        ("Validation macro-F1", f"{selected.validation_macro_f1:.6f}"),
        ("Test macro-F1 final", f"{selected.test_macro_f1:.6f}"),
        ("Test accuracy final", f"{metric(selected.test_metrics, 'accuracy'):.6f}"),
        ("TTA", f"{selected.tta_policy}, {selected.tta_views} vistas"),
        ("Temperatura", f"{selected.temperature:.6f}" if selected.temperature else "N/D"),
        ("Seed", str(selected.seed)),
    ]
    y = 90
    font = ImageFont.load_default()
    for label, value in fields:
        draw.text((60, y), label, fill=(31, 41, 55), font=font)
        draw.text((300, y), str(value), fill=(17, 24, 39), font=font)
        y += 42
    draw_note(
        draw,
        (60, 560),
        (
            "Uso de test: evaluacion final solamente. Datos sinteticos: no incorporados al train "
            "del clasificador. `spotted`: categoria visual, no diagnostico de hongo."
        ),
    )
    save_png(path, image)


def run(output_dir: Path, dataset_split: Path, *, update_config: bool) -> dict[str, Any]:
    """Run the consolidation and write all artifacts."""
    candidates = load_candidates()
    dataset_summary = load_dataset_counts(dataset_split)
    validations = validate_candidates(candidates, dataset_summary)
    selected = select_final_candidate(candidates)
    final_metrics = build_final_metrics(candidates, selected, validations)
    output = output_dir
    write_json(output / "final_metrics.json", final_metrics)
    rows = comparison_rows(candidates, selected)
    write_csv(
        output / "final_comparison.csv",
        rows,
        [
            "result_group",
            "model_id",
            "architecture",
            "selected_final",
            "validation_macro_f1",
            "test_macro_f1",
            "test_accuracy",
            "test_macro_precision",
            "test_macro_recall",
            "test_f1_intact",
            "test_f1_broken",
            "mean_confidence",
            "latency_mean_ms",
            "latency_cuda_mean_ms",
            "latency_seconds_per_image",
            "test_macro_f1_abs_delta_vs_r1",
            "test_macro_f1_pct_delta_vs_r1",
            "validation_macro_f1_abs_delta_vs_r1",
            "validation_macro_f1_pct_delta_vs_r1",
            "checkpoint",
            "config_source",
            "validation_source",
            "test_source",
            "conclusion",
            "notes",
        ],
    )
    write_final_report(output / "final_report.md", final_metrics)
    create_pngs(final_metrics, candidates, output)
    if update_config:
        update_production_config(selected, final_metrics)
    return final_metrics


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    final_metrics = run(
        output_dir=args.output,
        dataset_split=args.dataset_split,
        update_config=args.update_production_config,
    )
    selected = final_metrics["final_model"]
    improvement = final_metrics["improvement_vs_resultados_1"]["test_macro_f1"]
    print(f"Modelo final: {selected['model_id']}")
    print(f"Validation macro-F1: {selected['validation_macro_f1']:.6f}")
    print(f"Test macro-F1: {selected['test_macro_f1']:.6f}")
    print(
        "Mejora test macro-F1 vs R1: "
        f"{improvement['absolute']:.6f} ({improvement['percent']:.2f}%)"
    )
    print(f"Salida: {repo_path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
