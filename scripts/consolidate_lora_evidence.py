"""Consolidate reproducible evidence for the local SD 1.5 LoRA run.

The script reads only local configuration, metadata, notebook cells, manifests,
and existing files. It does not load model weights, run inference, or train.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "lora_sd15.yaml"
METADATA_PATH = REPO_ROOT / "data" / "lora" / "train" / "metadata.jsonl"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "06_entrenamiento_lora_sd15_colab.ipynb"
ADAPTER_PATH = REPO_ROOT / "models" / "lora" / "soybean_sd15" / "pytorch_lora_weights.safetensors"
RESULTS_DIR = REPO_ROOT / "results" / "lora"
SAMPLES_DIR = RESULTS_DIR / "samples"

CLASS_NAMES = ("intact", "broken", "spotted", "immature", "skin_damaged")
SAMPLE_LIMITS = {
    "intact": 2,
    "broken": 2,
    "spotted": 1,
    "immature": 1,
    "skin_damaged": 1,
}
PRIVATE_PATH_PATTERN = re.compile(
    r"([A-Za-z]:\\Users\\[^\\/\s]+)|(/Users/[^/\s]+)|(/home/[^/\s]+)"
)


@dataclass(frozen=True)
class EvidenceValue:
    """A verified value and its local evidence source."""

    value: Any
    source: str


def repo_path(path: Path) -> str:
    """Return a repository-relative path using POSIX separators."""

    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML from a local file."""

    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_json(path: Path) -> Any:
    """Load JSON from a local file."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    """Write pretty JSON with stable ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_scalar(value: str) -> Any:
    """Parse a scalar without inventing types beyond the literal text."""

    cleaned = value.strip().strip(",").strip()
    if (cleaned.startswith("'") and cleaned.endswith("'")) or (
        cleaned.startswith('"') and cleaned.endswith('"')
    ):
        cleaned = cleaned[1:-1]
    lowered = cleaned.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    if re.fullmatch(r"[-+]?\d+", cleaned):
        return int(cleaned)
    if re.fullmatch(r"[-+]?(\d*\.\d+|\d+e[-+]?\d+)", cleaned, flags=re.IGNORECASE):
        return float(cleaned)
    return cleaned


def read_metadata_records(path: Path) -> list[dict[str, Any]]:
    """Read metadata.jsonl records from disk."""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"metadata.jsonl line {line_number} is not a JSON object")
        records.append(record)
    return records


def class_from_record(record: dict[str, Any]) -> str | None:
    """Extract the visual class from a metadata record when it is explicit."""

    file_name = str(record.get("file_name", ""))
    text = str(record.get("text", ""))
    parts = [part.lower() for part in Path(file_name).parts]
    stem = Path(file_name).stem.lower()
    for class_name in sorted(CLASS_NAMES, key=len, reverse=True):
        if class_name in parts or stem.startswith(f"{class_name}_"):
            return class_name
    for class_name in sorted(CLASS_NAMES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(class_name)}\b", text.lower()):
            return class_name
    return None


def count_distribution(records: list[dict[str, Any]]) -> Counter[str]:
    """Count metadata records per explicit class."""

    distribution: Counter[str] = Counter()
    for record in records:
        class_name = class_from_record(record)
        if class_name is not None:
            distribution[class_name] += 1
    return distribution


def metadata_has_private_paths(records: list[dict[str, Any]]) -> bool:
    """Return True when metadata contains a private absolute path."""

    return any(PRIVATE_PATH_PATTERN.search(text) for record in records for text in iter_strings(record))


def iter_strings(value: Any) -> list[str]:
    """Collect string leaves from a nested value."""

    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for key, item in value.items():
            strings.extend(iter_strings(key))
            strings.extend(iter_strings(item))
        return strings
    if isinstance(value, list):
        strings = []
        for item in value:
            strings.extend(iter_strings(item))
        return strings
    return []


def notebook_text_blocks(path: Path) -> tuple[list[str], dict[str, Any]]:
    """Return all notebook source/output text blocks and execution summary."""

    if not path.exists():
        return [], {
            "exists": False,
            "total_cells": 0,
            "code_cells": 0,
            "executed_code_cells": 0,
            "cells_with_outputs": 0,
        }
    notebook = load_json(path)
    cells = notebook.get("cells", []) if isinstance(notebook, dict) else []
    blocks: list[str] = []
    executed = 0
    output_cells = 0
    code_cells = 0
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            blocks.append("".join(str(part) for part in source))
        elif isinstance(source, str):
            blocks.append(source)
        if cell.get("cell_type") == "code":
            code_cells += 1
            if cell.get("execution_count") is not None:
                executed += 1
        outputs = cell.get("outputs", [])
        if outputs:
            output_cells += 1
        for output in outputs:
            if not isinstance(output, dict):
                continue
            text = output.get("text")
            if isinstance(text, list):
                blocks.append("".join(str(part) for part in text))
            elif isinstance(text, str):
                blocks.append(text)
            data = output.get("data")
            if isinstance(data, dict):
                plain = data.get("text/plain")
                if isinstance(plain, list):
                    blocks.append("".join(str(part) for part in plain))
                elif isinstance(plain, str):
                    blocks.append(plain)
    summary = {
        "exists": True,
        "total_cells": len(cells),
        "code_cells": code_cells,
        "executed_code_cells": executed,
        "cells_with_outputs": output_cells,
    }
    return blocks, summary


def extract_notebook_parameters(path: Path) -> dict[str, EvidenceValue]:
    """Extract explicit training parameters from notebook sources and outputs."""

    key_map = {
        "pretrained_model_name_or_path": "base_model",
        "model_id": "base_model",
        "base_model": "base_model",
        "resolution": "resolution",
        "rank": "rank",
        "learning_rate": "learning_rate",
        "train_batch_size": "train_batch_size",
        "batch_size": "train_batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "max_train_steps": "max_train_steps",
        "seed": "seed",
        "trigger_word": "trigger_word",
        "mixed_precision": "mixed_precision",
    }
    blocks, _summary = notebook_text_blocks(path)
    found: dict[str, EvidenceValue] = {}
    assignment = re.compile(
        r"(?<![-\w])(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*"
        r"(?P<value>'[^'\n]*'|\"[^\"\n]*\"|[^\s,#]+)"
    )
    cli_flag = re.compile(r"--(?P<key>[A-Za-z0-9_-]+)(?:=|\s+)(?P<value>[^\s`\\]+)")
    for block in blocks:
        for match in assignment.finditer(block):
            raw_key = match.group("key").replace("-", "_")
            normalized = key_map.get(raw_key)
            if normalized and normalized not in found:
                found[normalized] = EvidenceValue(
                    parse_scalar(match.group("value")),
                    repo_path(path),
                )
        for match in cli_flag.finditer(block):
            raw_key = match.group("key").replace("-", "_")
            normalized = key_map.get(raw_key)
            if normalized and normalized not in found:
                found[normalized] = EvidenceValue(
                    parse_scalar(match.group("value")),
                    repo_path(path),
                )
    return found


def config_parameters(config: dict[str, Any]) -> dict[str, EvidenceValue]:
    """Extract configured LoRA parameters from configs/lora_sd15.yaml."""

    model = config.get("model", {}) if isinstance(config.get("model"), dict) else {}
    training = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    parameters: dict[str, EvidenceValue] = {}
    mapping = {
        "base_model": model.get("base_model"),
        "trigger_word": model.get("trigger_word"),
        "resolution": training.get("resolution"),
        "rank": training.get("rank"),
        "learning_rate": training.get("learning_rate"),
        "train_batch_size": training.get("train_batch_size"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "max_train_steps_initial": training.get("max_train_steps_initial"),
        "max_train_steps_full": training.get("max_train_steps_full"),
        "seed": training.get("seed"),
        "mixed_precision": training.get("mixed_precision"),
        "train_text_encoder": training.get("train_text_encoder"),
    }
    for key, value in mapping.items():
        if value is not None:
            parameters[key] = EvidenceValue(value, repo_path(CONFIG_PATH))
    return parameters


def adapter_info(path: Path) -> dict[str, Any]:
    """Return adapter file evidence without loading the model."""

    if not path.exists():
        return {"exists": False, "path": repo_path(path)}
    stat = path.stat()
    return {
        "exists": True,
        "path": repo_path(path),
        "file_name": path.name,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "last_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def write_dataset_distribution(path: Path, distribution: Counter[str]) -> None:
    """Write distribution by class as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_name", "count"])
        writer.writeheader()
        for class_name in CLASS_NAMES:
            writer.writerow({"class_name": class_name, "count": distribution.get(class_name, 0)})


def image_count(records: list[dict[str, Any]], metadata_path: Path) -> int:
    """Count image files referenced by metadata that currently exist."""

    base_dir = metadata_path.parent
    count = 0
    for record in records:
        file_name = record.get("file_name")
        if isinstance(file_name, str) and (base_dir / file_name).exists():
            count += 1
    return count


def copy_dataset_samples(records: list[dict[str, Any]], metadata_path: Path) -> list[dict[str, str]]:
    """Copy a small, deterministic sample selection from existing local images."""

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    selected: Counter[str] = Counter()
    copied: list[dict[str, str]] = []
    for record in records:
        class_name = class_from_record(record)
        if class_name is None or selected[class_name] >= SAMPLE_LIMITS[class_name]:
            continue
        file_name = record.get("file_name")
        if not isinstance(file_name, str):
            continue
        source = metadata_path.parent / file_name
        if not source.exists():
            continue
        destination = SAMPLES_DIR / f"{class_name}_{source.name}"
        shutil.copy2(source, destination)
        selected[class_name] += 1
        copied.append(
            {
                "class_name": class_name,
                "source": repo_path(source),
                "sample": repo_path(destination),
            }
        )
    return copied


def find_manifest_sources() -> list[Path]:
    """Find local manifests that may contain run evidence."""

    direct_candidates = [REPO_ROOT / "run_manifest.json"]
    search_roots = [RESULTS_DIR, REPO_ROOT / "models" / "lora", REPO_ROOT / "data" / "lora"]
    skipped_parts = {".git", ".venv", "venv", "__pycache__", "data/raw"}
    output_manifest = RESULTS_DIR / "run_manifest.json"
    found: list[Path] = [path for path in direct_candidates if path.exists()]
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*manifest*.json"):
            relative = repo_path(path)
            if any(part in relative for part in skipped_parts):
                continue
            if path.resolve() == output_manifest.resolve():
                continue
            found.append(path)
    unique = sorted({path.resolve(): path for path in found}.values(), key=repo_path)
    return unique


def find_comparison_images() -> list[Path]:
    """Find existing images whose names suggest base-vs-LoRA comparisons."""

    search_roots = [RESULTS_DIR, REPO_ROOT / "outputs", REPO_ROOT / "runs", REPO_ROOT / "models" / "lora"]
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    hint_pattern = re.compile(r"(base|lora|comparison|compare|vs)", flags=re.IGNORECASE)
    found: list[Path] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in image_extensions:
                relative = repo_path(path)
                if "results/lora/samples/" in relative:
                    continue
                if hint_pattern.search(relative):
                    found.append(path)
    return sorted(found, key=repo_path)


def comparison_manifest_rows(images: list[Path]) -> list[dict[str, str]]:
    """Build base-vs-LoRA manifest rows from existing images only."""

    rows: list[dict[str, str]] = []
    images_by_class: dict[str, list[Path]] = {class_name: [] for class_name in CLASS_NAMES}
    for image in images:
        relative = repo_path(image).lower()
        for class_name in sorted(CLASS_NAMES, key=len, reverse=True):
            if class_name in relative:
                images_by_class[class_name].append(image)
                break
    for class_name in CLASS_NAMES:
        class_images = images_by_class[class_name]
        base = next((path for path in class_images if "base" in repo_path(path).lower()), None)
        lora = next((path for path in class_images if "lora" in repo_path(path).lower()), None)
        comparison = next(
            (
                path
                for path in class_images
                if re.search(r"(comparison|compare|vs)", repo_path(path), flags=re.IGNORECASE)
            ),
            None,
        )
        evidence_status = "FOUND" if any([base, lora, comparison]) else "MISSING"
        rows.append(
            {
                "class_name": class_name,
                "prompt": "",
                "seed": "",
                "base_image": repo_path(base) if base else "",
                "lora_image": repo_path(lora) if lora else "",
                "comparison_image": repo_path(comparison) if comparison else "",
                "evidence_status": evidence_status,
            }
        )
    return rows


def write_comparison_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the comparison image manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_name",
        "prompt",
        "seed",
        "base_image",
        "lora_image",
        "comparison_image",
        "evidence_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def contains_private_path(value: Any) -> bool:
    """Detect private absolute paths in an arbitrary serializable value."""

    return any(PRIVATE_PATH_PATTERN.search(text) for text in iter_strings(value))


def write_report(
    path: Path,
    status: str,
    found: list[str],
    missing: list[str],
    parameters: dict[str, EvidenceValue],
    generated_files: list[str],
) -> None:
    """Write a compact Markdown evidence report."""

    lines = [
        "# LoRA SD1.5 evidence report",
        "",
        f"Status: **{status}**",
        "",
        "This report consolidates local evidence only. No retraining or inference was executed.",
        "",
        "## Evidence found",
    ]
    lines.extend(f"- {item}" for item in found)
    lines.extend(["", "## Evidence missing"])
    lines.extend(f"- {item}" for item in missing)
    lines.extend(["", "## Confirmed parameters"])
    for key in sorted(parameters):
        evidence = parameters[key]
        lines.append(f"- {key}: `{evidence.value}` ({evidence.source})")
    lines.extend(["", "## Generated files"])
    lines.extend(f"- {file_path}" for file_path in generated_files)
    lines.extend(
        [
            "",
            "## Notes",
            "- `spotted` is treated only as a visual category, not as a fungal diagnosis.",
            "- Synthetic images are not approved for classifier training unless human review accepts them.",
            "- LoRA weights are local artifacts and must not be versioned in Git.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def consolidate() -> dict[str, Any]:
    """Generate local LoRA evidence artifacts and return the summary."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_yaml(CONFIG_PATH)
    records = read_metadata_records(METADATA_PATH)
    distribution = count_distribution(records)
    notebook_parameters = extract_notebook_parameters(NOTEBOOK_PATH)
    notebook_blocks, notebook_summary = notebook_text_blocks(NOTEBOOK_PATH)
    parameters = config_parameters(config)
    for key, value in notebook_parameters.items():
        parameters.setdefault(key, value)

    adapter = adapter_info(ADAPTER_PATH)
    manifests = find_manifest_sources()
    comparison_images = find_comparison_images()
    comparison_rows = comparison_manifest_rows(comparison_images)
    copied_samples = copy_dataset_samples(records, METADATA_PATH)
    existing_image_count = image_count(records, METADATA_PATH)

    found = []
    missing = []
    if CONFIG_PATH.exists():
        found.append(f"Config: {repo_path(CONFIG_PATH)}")
    else:
        missing.append(f"Config file missing: {repo_path(CONFIG_PATH)}")
    if METADATA_PATH.exists():
        found.append(f"Metadata records: {len(records)} from {repo_path(METADATA_PATH)}")
    else:
        missing.append(f"Metadata file missing: {repo_path(METADATA_PATH)}")
    if adapter["exists"]:
        found.append(
            f"LoRA adapter: {adapter['file_name']} ({adapter['size_bytes']} bytes, "
            f"{adapter['size_mb']} MB)"
        )
    else:
        missing.append(f"LoRA adapter missing: {repo_path(ADAPTER_PATH)}")
    if NOTEBOOK_PATH.exists():
        found.append(
            f"Notebook: {repo_path(NOTEBOOK_PATH)}; executed code cells "
            f"{notebook_summary['executed_code_cells']}; cells with outputs "
            f"{notebook_summary['cells_with_outputs']}"
        )
    else:
        missing.append(f"Notebook missing: {repo_path(NOTEBOOK_PATH)}")
    if notebook_summary["cells_with_outputs"] == 0:
        missing.append("Notebook outputs/logs are missing; hardware and training time are not verified.")
    if not manifests:
        missing.append("No external run manifest was found.")
    else:
        found.extend(f"Manifest source: {repo_path(path)}" for path in manifests)
    if not comparison_images:
        missing.append("No base-vs-LoRA comparison images were found.")
    else:
        found.extend(f"Comparison image: {repo_path(path)}" for path in comparison_images)

    required_parameters = {
        "base_model",
        "resolution",
        "rank",
        "learning_rate",
        "train_batch_size",
        "gradient_accumulation_steps",
        "seed",
        "trigger_word",
    }
    for parameter in sorted(required_parameters - set(parameters)):
        missing.append(f"Parameter not verified: {parameter}")
    for parameter in ["hardware", "training_time"]:
        missing.append(f"Parameter not verified: {parameter}")

    status = "COMPLETE" if not missing else "PARTIAL"
    generated_files = [
        "results/lora/run_manifest.json",
        "results/lora/training_summary.json",
        "results/lora/training_config.yaml",
        "results/lora/dataset_distribution.csv",
        "results/lora/evidence_inventory.json",
        "results/lora/evidence_report.md",
        "results/lora/base_vs_lora_manifest.csv",
    ]

    write_dataset_distribution(RESULTS_DIR / "dataset_distribution.csv", distribution)
    write_comparison_manifest(RESULTS_DIR / "base_vs_lora_manifest.csv", comparison_rows)
    (RESULTS_DIR / "training_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    parameter_payload = {
        key: {"value": evidence.value, "source": evidence.source}
        for key, evidence in sorted(parameters.items())
    }
    run_manifest = {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "no_retraining_performed": True,
        "sources": {
            "config": repo_path(CONFIG_PATH),
            "metadata": repo_path(METADATA_PATH),
            "notebook": repo_path(NOTEBOOK_PATH),
            "adapter": repo_path(ADAPTER_PATH),
            "external_manifests": [repo_path(path) for path in manifests],
        },
        "confirmed_parameters": parameter_payload,
        "adapter": adapter,
        "dataset": {
            "metadata_records": len(records),
            "referenced_images_existing": existing_image_count,
            "class_distribution": {class_name: distribution.get(class_name, 0) for class_name in CLASS_NAMES},
            "metadata_private_paths_detected": metadata_has_private_paths(records),
        },
        "notebook": notebook_summary,
        "samples_copied": copied_samples,
        "comparison_images_found": [repo_path(path) for path in comparison_images],
        "missing_evidence": missing,
    }
    training_summary = {
        "status": status,
        "model_base": parameter_payload.get("base_model"),
        "adapter_file": adapter,
        "dataset_images": len(records),
        "dataset_images_existing": existing_image_count,
        "class_distribution": {class_name: distribution.get(class_name, 0) for class_name in CLASS_NAMES},
        "training_parameters": parameter_payload,
        "notebook_executed": notebook_summary["executed_code_cells"] > 0,
        "notebook_cells_with_outputs": notebook_summary["cells_with_outputs"],
        "no_retraining_performed": True,
    }
    evidence_inventory = {
        "status": status,
        "found": found,
        "missing": missing,
        "private_paths_detected": contains_private_path(run_manifest),
        "notebook_text_blocks_checked": len(notebook_blocks),
    }

    write_json(RESULTS_DIR / "run_manifest.json", run_manifest)
    write_json(RESULTS_DIR / "training_summary.json", training_summary)
    write_json(RESULTS_DIR / "evidence_inventory.json", evidence_inventory)
    write_report(
        RESULTS_DIR / "evidence_report.md",
        status,
        found,
        missing,
        parameters,
        generated_files + [sample["sample"] for sample in copied_samples],
    )
    return training_summary


def main() -> None:
    """CLI entrypoint."""

    summary = consolidate()
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
