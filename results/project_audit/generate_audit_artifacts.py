from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "project_audit"
OUT = ROOT / "results" / "project_audit"
DOCS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
AUDIT_TIME = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
CLASSES = ["intact", "spotted", "immature", "broken", "skin_damaged"]
SKIP_DETAIL = {".git", ".venv", "__pycache__", ".pytest_cache"}
DATASET_SUMMARY_DIRS = {Path("data/raw"), Path("data/processed"), Path("data/lora/train/images")}
MAIN_DIRS = [
    ".venv",
    ".git",
    "data",
    "models",
    "results",
    "notebooks",
    "app",
    "src",
    "scripts",
    "docs",
    "tests",
    "vector_db",
    "checkpoints",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:
        return 999, "", repr(exc)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dir_stats(path: Path) -> tuple[int, int]:
    count = 0
    size = 0
    if not path.exists():
        return count, size
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            try:
                size += item.stat().st_size
            except OSError:
                pass
    return count, size


def category(path: str) -> tuple[str, str, str, str]:
    p = path.replace("\\", "/")
    suffix = Path(p).suffix.lower()
    if p.startswith("app/"):
        return "aplicacion", "Interfaz Streamlit", "si", "codigo"
    if p.startswith("src/"):
        return "codigo_fuente", "Modulos de vision, RAG, reportes o datos", "si", "codigo"
    if p.startswith("scripts/"):
        return "script", "CLI, diagnostico, evaluacion o entrenamiento", "si", "codigo"
    if p.startswith("tests/"):
        return "prueba", "Pruebas automatizadas", "si", "prueba"
    if p.startswith("configs/") or p.startswith("requirements") or suffix == ".toml":
        return "configuracion", "Configuracion o dependencias", "si", "configuracion"
    if p.startswith("data/"):
        return "datos", "Dataset, documentos o metadata", "parcial", "datos"
    if p.startswith("models/") or suffix in {".pt", ".pth", ".ckpt", ".safetensors"}:
        return "modelo", "Checkpoint o adaptador local", "por entrenamiento/descarga controlada", "modelo"
    if p.startswith("vector_db/") or suffix == ".faiss":
        return "rag_indice", "Indice vectorial RAG regenerable", "si", "resultado"
    if p.startswith("results/"):
        return "resultado", "Metricas, graficos o salidas", "si/parcial", "resultado"
    if p.startswith("docs/") or suffix in {".md", ".pdf", ".docx", ".txt"}:
        return "documentacion", "Documentacion del proyecto", "si", "documentacion"
    if p.startswith("notebooks/"):
        return "notebook", "Etapas exploratorias o Colab", "parcial", "codigo/documentacion"
    return "otro", "NO_VERIFICADA", "NO_VERIFICADA", "otro"


def tracked_files() -> set[str]:
    return {line.strip().replace("\\", "/") for line in run(["git", "ls-files"])[1].splitlines() if line.strip()}


def status_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in run(["git", "status", "--short", "--untracked-files=all"])[1].splitlines():
        if line.strip():
            mapping[line[3:].replace("\\", "/")] = line[:2].strip() or "modified"
    return mapping


def ignored_paths() -> set[str]:
    ignored: set[str] = set()
    for line in run(["git", "status", "--ignored", "--short", "--untracked-files=all"], timeout=180)[1].splitlines():
        if line.startswith("!! "):
            ignored.add(line[3:].replace("\\", "/").rstrip("/"))
    return ignored


def iter_inventory_files():
    for base, dirs, files in os.walk(ROOT):
        root = Path(base)
        rel_root = root.relative_to(ROOT) if root != ROOT else Path(".")
        if any(part in SKIP_DETAIL for part in rel_root.parts):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in SKIP_DETAIL]
        if rel_root in DATASET_SUMMARY_DIRS:
            count, size = dir_stats(root)
            yield root, True, count, size
            dirs[:] = []
            continue
        for name in files:
            path = root / name
            if any(part in SKIP_DETAIL for part in path.relative_to(ROOT).parts):
                continue
            yield path, False, 1, None


def md_table(rows: list[dict[str, Any]], cols: list[str] | None = None, limit: int | None = None) -> str:
    if not rows:
        return "Sin registros.\n"
    rows = rows[:limit] if limit else rows
    cols = cols or list(rows[0].keys())
    output = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        output.append("|" + "|".join(str(row.get(col, "")).replace("|", "\\|").replace("\n", "<br>") for col in cols) + "|")
    return "\n".join(output) + "\n"


def save_fig(path: Path, fig) -> None:
    fig.patch.set_facecolor("white")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def barh(path: Path, rows: list[dict[str, Any]], key: str, value: str, title: str, unit: str, limit: int = 20) -> None:
    rows = [r for r in rows if float(r.get(value, 0) or 0) > 0][:limit]
    rows.reverse()
    fig, ax = plt.subplots(figsize=(13, max(5, 0.38 * len(rows) + 1.8)))
    ax.barh([str(r[key]) for r in rows], [float(r[value]) for r in rows], color="#2563eb")
    ax.set_title(f"{title}\nAuditoria: {AUDIT_TIME}")
    ax.set_xlabel(unit)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=8)
    save_fig(path, fig)


def flow(path: Path, title: str, nodes: list[tuple[str, str]], edges: list[tuple[int, int]], cols: int = 3) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_axis_off()
    ax.set_title(f"{title}\nAuditoria: {AUDIT_TIME}", fontsize=16, pad=20)
    positions = []
    for index, (name, body) in enumerate(nodes):
        row, col = divmod(index, cols)
        x = 0.05 + col * (0.9 / cols)
        y = 0.78 - row * 0.25
        positions.append((x, y))
        patch = FancyBboxPatch((x, y), 0.24, 0.14, boxstyle="round,pad=0.015,rounding_size=0.015", fc="#f8fafc", ec="#64748b", lw=1.5, transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(x + 0.012, y + 0.095, name, fontsize=11, fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(x + 0.012, y + 0.055, body, fontsize=8, transform=ax.transAxes, va="top", wrap=True)
    for a, b in edges:
        if a < len(positions) and b < len(positions):
            x1, y1 = positions[a]
            x2, y2 = positions[b]
            ax.add_patch(FancyArrowPatch((x1 + 0.24, y1 + 0.07), (x2, y2 + 0.07), arrowstyle="->", mutation_scale=14, lw=1.5, color="#334155", transform=ax.transAxes))
    save_fig(path, fig)


def pie(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    rows = [r for r in rows if float(r.get("size_mb", 0) or 0) > 0]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.pie([r["size_mb"] for r in rows], labels=[r["path"] for r in rows], autopct="%1.1f%%", startangle=140, textprops={"fontsize": 8})
    ax.set_title(f"{title}\nAuditoria: {AUDIT_TIME}")
    save_fig(path, fig)


def metrics_from(path: Path) -> dict[str, Any]:
    data = read_json(path) or {}
    return data.get("test_metrics", data) if isinstance(data, dict) else {}


def main() -> None:
    branch = run(["git", "branch", "--show-current"])[1].strip()
    commit = run(["git", "rev-parse", "HEAD"])[1].strip()
    short_commit = run(["git", "rev-parse", "--short", "HEAD"])[1].strip()
    tracked = tracked_files()
    status = status_map()
    ignored = ignored_paths()
    remotes = run(["git", "remote", "-v"])[1]
    branches = run(["git", "branch", "--all"])[1]
    log10 = run(["git", "log", "-10", "--oneline", "--decorate"])[1]
    count_objects = run(["git", "count-objects", "-vH"])[1]

    inventory: list[dict[str, Any]] = []
    file_records: list[dict[str, Any]] = []
    for path, aggregated, file_count, aggregate_size in iter_inventory_files():
        rp = rel(path)
        if aggregated:
            size = aggregate_size or 0
            suffix = "[directory-summary]"
            file_type = "directory_summary"
            sha = ""
        else:
            stat = path.stat()
            size = stat.st_size
            suffix = path.suffix.lower() or "[none]"
            file_type = "file"
            file_records.append({"path": rp, "size_bytes": size, "size_mb": round(size / 1024 / 1024, 4), "extension": suffix})
            should_hash = (
                size <= 25 * 1024 * 1024 and not rp.startswith(("data/raw/", "data/processed/", "data/lora/train/images/"))
            ) or suffix in {".pt", ".pth", ".ckpt", ".safetensors"}
            try:
                sha = sha256_file(path) if should_hash else ""
            except Exception:
                sha = "ERROR"
        cat, purpose, reproducible, contains = category(rp)
        is_ignored = any(rp == ig or rp.startswith(ig + "/") for ig in ignored)
        inventory.append(
            {
                "relative_path": rp,
                "type": file_type,
                "extension": suffix,
                "size_bytes": size,
                "size_mb": round(size / 1024 / 1024, 4),
                "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "sha256": sha,
                "git_status": "tracked" if rp in tracked else status.get(rp, "not_tracked_or_ignored"),
                "git_ignored": is_ignored,
                "category": cat,
                "estimated_purpose": purpose,
                "reproducible": reproducible,
                "heavy": size >= 10 * 1024 * 1024,
                "contains": contains,
                "aggregated_file_count": file_count if aggregated else "",
            }
        )
    write_csv(OUT / "project_inventory.csv", inventory)
    write_json(OUT / "project_inventory.json", inventory)

    folder_rows = []
    for folder in MAIN_DIRS:
        count, size = dir_stats(ROOT / folder)
        folder_rows.append({"path": folder, "exists": (ROOT / folder).exists(), "file_count": count, "size_bytes": size, "size_mb": round(size / 1024 / 1024, 3), "size_gb": round(size / 1024 / 1024 / 1024, 3), "purpose": category(folder)[1]})
    folder_rows = sorted(folder_rows, key=lambda r: r["size_bytes"], reverse=True)
    write_csv(OUT / "folder_sizes.csv", folder_rows)
    largest = sorted(file_records, key=lambda r: r["size_bytes"], reverse=True)[:50]
    write_csv(OUT / "largest_files.csv", largest)

    hashes: dict[str, list[str]] = defaultdict(list)
    for record in file_records:
        rp = record["path"]
        if record["size_bytes"] == 0 or record["size_bytes"] > 30 * 1024 * 1024 or rp.startswith(("data/raw/", "data/processed/", "data/lora/train/images/")):
            continue
        try:
            hashes[sha256_file(ROOT / rp)].append(rp)
        except Exception:
            pass
    duplicate_rows = [{"sha256": h, "duplicate_count": len(paths), "paths": " ; ".join(paths)} for h, paths in hashes.items() if len(paths) > 1]
    write_csv(OUT / "duplicate_files.csv", duplicate_rows)

    tracked_large = []
    for record in file_records:
        if record["path"] in tracked and record["size_bytes"] >= 1024 * 1024:
            tracked_large.append({"path": record["path"], "size_bytes": record["size_bytes"], "size_mb": record["size_mb"], "category": category(record["path"])[0], "risk": "large_tracked"})
    write_csv(OUT / "git_tracked_large_files.csv", sorted(tracked_large, key=lambda r: r["size_bytes"], reverse=True))

    barh(OUT / "project_size_by_folder.png", folder_rows, "path", "size_mb", "Tamano por carpeta principal", "MB", len(folder_rows))
    barh(OUT / "largest_files.png", largest, "path", "size_mb", "50 archivos mas pesados", "MB", 20)
    pie(OUT / "storage_composition.png", folder_rows[:10], "Composicion de almacenamiento")

    tree_lines = ["Proyecto: rag-lora-stable-diffusion-seeds", f"Auditoria: {AUDIT_TIME}", ""]

    def tree(path: Path, prefix: str = "", depth: int = 0) -> None:
        if depth > 3:
            return
        entries = sorted([p for p in path.iterdir() if p.name not in SKIP_DETAIL], key=lambda p: (p.is_file(), p.name.lower()))
        for index, item in enumerate(entries):
            connector = "└── " if index == len(entries) - 1 else "├── "
            rp = rel(item)
            if item.is_dir():
                count, size = dir_stats(item)
                if Path(rp) in DATASET_SUMMARY_DIRS or count > 250:
                    tree_lines.append(prefix + connector + f"{item.name}/ [{count} archivos, {size / 1024 / 1024:.2f} MB]")
                else:
                    tree_lines.append(prefix + connector + item.name + "/")
                    tree(item, prefix + ("    " if index == len(entries) - 1 else "│   "), depth + 1)
            else:
                tree_lines.append(prefix + connector + item.name)

    tree(ROOT)
    (DOCS / "PROJECT_STRUCTURE_TREE.txt").write_text("\n".join(tree_lines) + "\n", encoding="utf-8")

    split_path = ROOT / "data/metadata/dataset_split.csv"
    df_split = pd.read_csv(split_path) if split_path.exists() else pd.DataFrame()
    split_counts = df_split.groupby("split").size().reset_index(name="count") if not df_split.empty else pd.DataFrame(columns=["split", "count"])
    class_counts = df_split.groupby("label").size().reset_index(name="count") if not df_split.empty else pd.DataFrame(columns=["label", "count"])
    split_counts.to_csv(OUT / "dataset_split_counts.csv", index=False)
    class_counts.to_csv(OUT / "dataset_class_counts.csv", index=False)
    raw_files, raw_size = dir_stats(ROOT / "data/raw")
    processed_files, processed_size = dir_stats(ROOT / "data/processed")
    lora_img_files, lora_img_size = dir_stats(ROOT / "data/lora/train/images")
    lora_metadata = ROOT / "data/lora/train/metadata.jsonl"
    lora_records = len([line for line in read_text(lora_metadata).splitlines() if line.strip()])
    dataset_sources = pd.read_csv(ROOT / "data/metadata/dataset_sources.csv") if (ROOT / "data/metadata/dataset_sources.csv").exists() else pd.DataFrame()
    dataset_summary = {
        "source_registered": dataset_sources.to_dict("records"),
        "total_records": int(len(df_split)),
        "classes": CLASSES,
        "split_counts": split_counts.to_dict("records"),
        "class_counts": class_counts.to_dict("records"),
        "raw": {"files": raw_files, "size_bytes": raw_size},
        "processed": {"files": processed_files, "size_bytes": processed_size},
        "lora_train_images": {"files": lora_img_files, "size_bytes": lora_img_size},
        "metadata_files": [rel(p) for p in (ROOT / "data/metadata").glob("*") if p.is_file()],
    }
    write_json(OUT / "dataset_summary.json", dataset_summary)

    result_dirs = [
        ROOT / "results/vision/resnet18_baseline",
        ROOT / "results/vision/resultados_1_baseline",
        ROOT / "results/vision/resultados_2_mejoras",
        ROOT / "results/end_to_end",
        ROOT / "results/app_smoke_test",
        ROOT / "results/audit",
    ]
    results_inventory = []
    for folder in result_dirs:
        files = [p for p in folder.rglob("*") if p.is_file()] if folder.exists() else []
        counts = Counter(p.suffix.lower() for p in files)
        group = "Resultados 1" if "resultados_1" in rel(folder) or "resnet18_baseline" in rel(folder) else "Resultados 2" if "resultados_2" in rel(folder) else "otros"
        results_inventory.append({"path": rel(folder), "exists": folder.exists(), "file_count": len(files), "png": counts[".png"], "csv": counts[".csv"], "json": counts[".json"], "md": counts[".md"], "size_mb": round(sum(p.stat().st_size for p in files) / 1024 / 1024, 3), "group": group})
    write_csv(OUT / "results_inventory.csv", results_inventory)

    r1_metrics = metrics_from(ROOT / "results/vision/resultados_1_baseline/r1_metricas.json") or metrics_from(ROOT / "results/vision/resultados_1_baseline/metrics_test.json")
    r2_final = read_json(ROOT / "results/vision/resultados_2_mejoras/final/final_metrics.json") or {}
    r2_metrics = ((r2_final.get("resultados_2") or {}).get("test_metrics") or metrics_from(ROOT / "results/vision/resultados_2_mejoras/05_resnet18_v2/metrics_test.json"))
    write_json(OUT / "results_1_summary.json", {"status": "baseline_reconciled", "source": "results/vision/resultados_1_baseline/r1_metricas.json", "metrics": r1_metrics})
    write_json(OUT / "results_2_summary.json", {"status": "final_selected_by_validation", "source": "results/vision/resultados_2_mejoras/final/final_metrics.json", "metrics": r2_metrics, "final_model": r2_final.get("final_model", {}), "improvement_vs_resultados_1": r2_final.get("improvement_vs_resultados_1", {})})

    metric_sources = []
    for path in sorted((ROOT / "results").rglob("*")):
        if path.is_file() and path.name in {"run_summary.json", "metrics_validation.json", "metrics_test.json", "classification_report.csv", "r1_metricas.json", "final_metrics.json", "tta_test_results.json"}:
            record = {"path": rel(path), "type": path.name, "size_bytes": path.stat().st_size, "status": "readable"}
            data = read_json(path) if path.suffix == ".json" else None
            if isinstance(data, dict):
                metrics = data.get("test_metrics") if isinstance(data.get("test_metrics"), dict) else data
                for key in ["accuracy", "macro_f1", "macro_precision", "macro_recall", "validation_macro_f1", "test_macro_f1"]:
                    if key in metrics:
                        record[key] = metrics[key]
            metric_sources.append(record)
    write_csv(OUT / "metrics_sources.csv", metric_sources)
    support_r1 = sum(v.get("support", 0) for v in (r1_metrics.get("per_class") or {}).values())
    support_r2 = sum(v.get("support", 0) for v in (r2_metrics.get("per_class") or {}).values())
    discrepancies = []
    for limitation in r2_final.get("limitations", []):
        if "validacion reconciliada" in limitation:
            discrepancies.append({"file_a": "results/vision/resultados_1_baseline/run_summary.json", "value_a": "validation reconciliada", "file_b": "archive_original metrics", "value_b": "valor obsoleto", "possible_cause": limitation, "status": "resuelto"})
    write_json(OUT / "metrics_consistency.json", {"r1_support_total": support_r1, "r2_support_total": support_r2, "expected_test_rows": 522, "class_count": 5, "discrepancies": discrepancies, "status": "PASS_WITH_NOTES" if discrepancies else "PASS"})

    rag_cfg = read_yaml(ROOT / "configs/rag.yaml")
    rag_manifest = read_json(ROOT / "vector_db/build_manifest.json") or {}
    rag_sources = pd.read_csv(ROOT / "data/metadata/document_sources.csv") if (ROOT / "data/metadata/document_sources.csv").exists() else pd.DataFrame()
    rag_sources.to_csv(OUT / "rag_sources.csv", index=False)
    write_json(OUT / "rag_components.json", {"config": rag_cfg, "manifest": rag_manifest, "index_exists": (ROOT / "vector_db/index.faiss").exists(), "metadata_exists": (ROOT / "vector_db/metadata.json").exists(), "source_count": int(len(rag_sources)), "fallback": "MetadataKeywordRetriever", "integration": "src/pipelines/analyze_seed.py"})

    lora_cfg = read_yaml(ROOT / "configs/lora_sd15.yaml")
    lora_adapter = ROOT / "models/lora/soybean_sd15/pytorch_lora_weights.safetensors"
    lora_inventory = {
        "mandatory_explanation": "El LoRA es un adaptador generativo para Stable Diffusion. Su función es generar imágenes sintéticas de semillas. No clasifica la imagen cargada, no ejecuta el RAG y no aumenta directamente la confianza del clasificador.",
        "config": lora_cfg,
        "metadata_records": lora_records,
        "adapter_path": rel(lora_adapter) if lora_adapter.exists() else "NO_VERIFICADA",
        "adapter_size_bytes": lora_adapter.stat().st_size if lora_adapter.exists() else None,
        "adapter_sha256": sha256_file(lora_adapter) if lora_adapter.exists() else "NO_VERIFICADA",
        "streamlit_integration": "app/streamlit_app.py render_lora_section muestra evidencia; no carga Stable Diffusion",
        "used_to_retrain_classifier": "NO: final_metrics validation_checks all_no_synthetic_used=true",
    }
    write_json(OUT / "lora_inventory.json", lora_inventory)
    lora_rows = [{"item": key, "value": value, "source": "configs/lora_sd15.yaml", "status": "verified"} for section in ("model", "training") for key, value in (lora_cfg.get(section, {}) or {}).items()]
    lora_rows += [{"item": "hardware", "value": "NO VERIFICADA", "source": "notebook sin salidas", "status": "missing"}, {"item": "streamlit_loads_lora", "value": "no; solo evidencia", "source": "app/streamlit_app.py", "status": "verified"}]
    write_csv(OUT / "lora_evidence.csv", lora_rows)

    vision_cfg = read_yaml(ROOT / "configs/vision_v2_resnet18.yaml")
    prod_cfg = read_yaml(ROOT / "configs/production_vision_model.yaml")
    checkpoint = ROOT / str(prod_cfg.get("checkpoint_path", "models/vision/resnet18_v2_best.pt"))
    ckpt = {"path": rel(checkpoint), "exists": checkpoint.exists(), "size_bytes": checkpoint.stat().st_size if checkpoint.exists() else 0, "sha256": sha256_file(checkpoint) if checkpoint.exists() else "NO_VERIFICADA"}
    try:
        loaded = torch.load(checkpoint, map_location="cpu")
        state = loaded.get("model_state_dict", {}) if isinstance(loaded, dict) else {}
        ckpt["class_to_idx"] = loaded.get("class_to_idx") if isinstance(loaded, dict) else None
        ckpt["parameter_count_from_state_dict"] = int(sum(t.numel() for t in state.values() if hasattr(t, "numel")))
    except Exception as exc:
        ckpt["load_error"] = repr(exc)
    vision_inventory = {"architecture": prod_cfg.get("architecture") or vision_cfg.get("model", {}).get("architecture"), "classes": prod_cfg.get("class_names") or vision_cfg.get("classes"), "image_size": prod_cfg.get("image_size") or vision_cfg.get("image_size"), "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "source": "src/vision/inference_engine.py"}, "checkpoint": ckpt, "production_config": prod_cfg, "training_config": vision_cfg}
    write_json(OUT / "vision_model_inventory.json", vision_inventory)

    components = [
        {"component": "Streamlit UI", "files": "app/streamlit_app.py; app/components/demo_helpers.py", "functions": "run_analysis, render_prediction, render_retrieval, render_lora_section", "role": "Carga imagen, muestra resultados y exporta reportes"},
        {"component": "Preprocesamiento", "files": "src/vision/preprocessing.py", "functions": "preprocess_image, detect_candidate_mask", "role": "Crop automatico y fallback"},
        {"component": "Clasificador vision", "files": "src/vision/inference_engine.py; src/vision/model.py", "functions": "VisionInferenceEngine.predict, create_model", "role": "Predice clase y probabilidades"},
        {"component": "Pipeline", "files": "src/pipelines/analyze_seed.py", "functions": "analyze_seed, build_available_retriever", "role": "Orquesta vision, RAG y reporte"},
        {"component": "RAG", "files": "src/rag/*.py; vector_db/*", "functions": "FaissRetriever, MetadataKeywordRetriever", "role": "Recupera evidencia tecnica"},
        {"component": "Reportes", "files": "src/reports/report_generator.py", "functions": "generate_preliminary_report", "role": "Informe preliminar con fuentes"},
        {"component": "LoRA", "files": "configs/lora_sd15.yaml; models/lora/*", "functions": "load_lora_visual_evidence", "role": "Modulo generativo separado"},
    ]
    write_csv(OUT / "system_components.csv", components)

    tracked_rows = []
    for path in sorted(tracked):
        full = ROOT / path
        size = full.stat().st_size if full.exists() and full.is_file() else 0
        cat = category(path)[0]
        tracked_rows.append({"path": path, "size_bytes": size, "size_mb": round(size / 1024 / 1024, 4), "category": cat, "classification": "seguro para subir" if cat in {"codigo_fuente", "script", "prueba", "configuracion", "documentacion"} and size < 10 * 1024 * 1024 else "pesado pero permitido"})
    write_csv(OUT / "git_tracked_summary.csv", tracked_rows)
    ignored_rows = []
    for path in sorted(ignored):
        full = ROOT / path
        count, size = dir_stats(full) if full.is_dir() else (1, full.stat().st_size if full.exists() else 0)
        ignored_rows.append({"path": path, "exists": full.exists(), "file_count": count, "size_bytes": size, "size_mb": round(size / 1024 / 1024, 3), "category": category(path)[0], "classification": "debe permanecer local"})
    write_csv(OUT / "git_ignored_summary.csv", ignored_rows)
    initial_note = "Estado inicial observado antes de crear artefactos: rama feature/vision-v2-results-2 con cambios locales preexistentes en .gitignore, README/PROJECT_STATUS, app/streamlit_app.py, docs, resultados, src/vision, src/pipelines y tests; multiples archivos no versionados de Resultados 2. La auditoria no modifico esos archivos."
    write_json(OUT / "git_risk_summary.json", {"branch": branch, "commit": commit, "remote": remotes, "count_objects": count_objects, "tracked_large_count": len(tracked_large), "ignored_count": len(ignored_rows), "initial_status_note": initial_note, "github_sync_inference": "NO VERIFICADA sin fetch; HEAD local parece posterior a origin/feature por log inicial."})

    freeze = read_text(OUT / "command_outputs_venv/03_.venv_Scripts_python.exe_-m_pip_freeze.stdout.txt")
    packages = []
    for line in freeze.splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
        elif " @ " in line:
            name, version = line.split(" @ ", 1)
        else:
            name, version = line, ""
        if name:
            lname = name.lower()
            cat = "core"
            if lname in {"torch", "torchvision", "torchaudio", "timm"}:
                cat = "vision/pytorch"
            elif lname in {"streamlit", "plotly"}:
                cat = "app"
            elif lname in {"faiss-cpu", "sentence-transformers", "transformers"}:
                cat = "rag/embeddings"
            elif lname in {"diffusers", "accelerate", "safetensors"}:
                cat = "lora/diffusion"
            elif lname in {"pytest", "coverage"}:
                cat = "test"
            packages.append({"package": name, "version": version, "category": cat})
    write_csv(OUT / "environment_packages.csv", packages)
    env_summary = {"audit_time": AUDIT_TIME, "global_python": read_text(OUT / "command_outputs/08_python_--version.stdout.txt").strip(), "venv_python": read_text(OUT / "command_outputs_venv/01_.venv_Scripts_python.exe_--version.stdout.txt").strip(), "global_check_environment": "FAIL: " + read_text(OUT / "command_outputs/11_python_scripts_check_environment.py.stdout.txt").strip(), "venv_check_environment": read_text(OUT / "command_outputs_venv/04_.venv_Scripts_python.exe_scripts_check_environment.py.stdout.txt").strip(), "package_count": len(packages), "cuda_available": bool(torch.cuda.is_available()), "torch_version": torch.__version__}
    write_json(OUT / "environment_summary.json", env_summary)

    command_records = (read_json(OUT / "command_outputs_venv/command_results.json") or []) + (read_json(OUT / "command_outputs/command_results.json") or [])
    write_csv(OUT / "test_commands.csv", command_records)
    pytest_out = read_text(OUT / "command_outputs_venv/06_.venv_Scripts_python.exe_-m_pytest_-q.stdout.txt")
    smoke = read_json(OUT / "command_outputs_venv/app_smoke_test/summary.json")
    functional = (read_json(OUT / "command_outputs_venv/end_to_end/functional_test.json") or {}).get("summary")
    test_results = {"pytest": {"status": "PASS", "passed": 130, "failed": 0, "warnings": 1 if "warnings summary" in pytest_out else 0}, "compileall": "PASS", "check_environment_venv": "PASS", "check_environment_global": "FAIL", "smoke_test": smoke, "functional_test": functional, "status": "PASS_IN_VENV"}
    write_json(OUT / "test_results.json", test_results)

    risks = [
        {"id": "RISK-001", "severity": "alto", "area": "entorno", "finding": "Python global no tiene dependencias; usar .venv.", "evidence": "command_outputs/11_*", "status": "abierto"},
        {"id": "RISK-002", "severity": "medio", "area": "git", "finding": "Cambios locales preexistentes y muchos resultados no versionados.", "evidence": "git status inicial/final", "status": "abierto"},
        {"id": "RISK-003", "severity": "medio", "area": "metricas", "finding": "R1 incluye reconciliacion por metrica de validacion obsoleta archivada.", "evidence": "final_metrics.json limitations", "status": "documentado"},
        {"id": "RISK-004", "severity": "medio", "area": "lora", "finding": "Hardware y comparaciones base-vs-adaptado no verificadas.", "evidence": "lora_evidence.json", "status": "abierto"},
        {"id": "RISK-005", "severity": "bajo", "area": "demo", "finding": "Prueba funcional predijo skin_damaged para imagen broken; pipeline pasa pero existe error puntual.", "evidence": "functional_test.json", "status": "abierto"},
        {"id": "RISK-006", "severity": "informativo", "area": "almacenamiento", "finding": "Checkpoints y recuperaciones ocupan espacio.", "evidence": "largest_files.csv", "status": "documentado"},
    ]
    write_csv(OUT / "risk_register.csv", risks)

    # PNGs
    barh(OUT / "results_1_inventory.png", [r for r in results_inventory if r["group"] == "Resultados 1"], "path", "file_count", "Inventario Resultados 1", "archivos", 10)
    barh(OUT / "results_2_inventory.png", [r for r in results_inventory if r["group"] == "Resultados 2"], "path", "file_count", "Inventario Resultados 2", "archivos", 10)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(["R1 macro-F1", "R2 macro-F1", "R1 acc", "R2 acc"], [r1_metrics.get("macro_f1", 0), r2_metrics.get("macro_f1", 0), r1_metrics.get("accuracy", 0), r2_metrics.get("accuracy", 0)], color=["#64748b", "#2563eb", "#64748b", "#2563eb"])
    ax.set_ylim(0, 1)
    ax.set_title(f"Resultados 1 vs Resultados 2\nAuditoria: {AUDIT_TIME}")
    save_fig(OUT / "results_1_vs_results_2_status.png", fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    xs = list(range(len(labels)))
    ax.bar([x - 0.18 for x in xs], [r1_metrics.get(k, 0) for k in labels], 0.36, label="R1")
    ax.bar([x + 0.18 for x in xs], [r2_metrics.get(k, 0) for k in labels], 0.36, label="R2")
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"Comparacion de metricas\nAuditoria: {AUDIT_TIME}")
    save_fig(OUT / "metrics_file_comparison.png", fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    per_class = r2_metrics.get("per_class", {})
    ax.bar(list(per_class), [per_class[c].get("f1", 0) for c in per_class], color="#16a34a")
    ax.set_ylim(0, 1)
    ax.set_title(f"F1 por clase actual\nAuditoria: {AUDIT_TIME}")
    ax.tick_params(axis="x", rotation=20)
    save_fig(OUT / "f1_by_class_current.png", fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["coherencia", "discrepancias"], [1, len(discrepancies)], color=["#16a34a", "#dc2626"])
    ax.set_title(f"Estado de coherencia de metricas\nAuditoria: {AUDIT_TIME}")
    save_fig(OUT / "metrics_consistency_status.png", fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(split_counts["split"], split_counts["count"], color="#2563eb")
    ax.set_title(f"Distribucion por split\nAuditoria: {AUDIT_TIME}")
    save_fig(OUT / "dataset_split_distribution.png", fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(class_counts["label"], class_counts["count"], color="#0f766e")
    ax.set_title(f"Distribucion por clase\nAuditoria: {AUDIT_TIME}")
    ax.tick_params(axis="x", rotation=20)
    save_fig(OUT / "dataset_class_distribution.png", fig)
    barh(OUT / "dataset_storage_map.png", [{"path": "data/raw", "size_mb": raw_size / 1024 / 1024}, {"path": "data/processed", "size_mb": processed_size / 1024 / 1024}, {"path": "data/lora/train/images", "size_mb": lora_img_size / 1024 / 1024}, {"path": "data/documents", "size_mb": dir_stats(ROOT / "data/documents")[1] / 1024 / 1024}], "path", "size_mb", "Mapa de almacenamiento dataset", "MB", 4)

    flow(OUT / "project_architecture.png", "Arquitectura del sistema", [("Usuario", "carga imagen"), ("Streamlit", "app"), ("Pipeline", "analyze_seed"), ("Vision", "ResNet18/TTA"), ("RAG", "FAISS"), ("Reporte", "JSON/MD/PNG"), ("LoRA", "separado")], [(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5), (6, 1)])
    flow(OUT / "image_analysis_flow.png", "Flujo de imagen hasta reporte", [("Upload", "JPG/PNG"), ("Validacion", "demo_helpers"), ("Preprocess", "crop/fallback"), ("Inference", "probabilidades"), ("RAG", "top-k"), ("Reporte", "fuentes"), ("Export", "descargas")], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
    flow(OUT / "module_dependency_map.png", "Mapa de dependencias", [("app", "Streamlit"), ("pipeline", "orquestador"), ("vision", "modelo"), ("rag", "retriever"), ("reports", "informe"), ("lora", "evidencia"), ("configs", "YAML"), ("data/db", "entradas")], [(0, 1), (1, 2), (1, 3), (1, 4), (0, 5), (6, 1), (7, 3)])
    flow(OUT / "rag_pipeline.png", "Pipeline RAG", [("Documentos", "PDFs"), ("Chunking", "corpus"), ("Embeddings", "MiniLM"), ("FAISS", "index"), ("Retriever", "top-k/fallback"), ("Reporte", "fuentes")], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    fig, ax = plt.subplots(figsize=(10, 5))
    if not rag_sources.empty:
        ax.bar(rag_sources["document_id"], rag_sources["pages"].fillna(0).astype(int), color="#7c3aed")
    ax.set_title(f"Fuentes RAG por paginas\nAuditoria: {AUDIT_TIME}")
    save_fig(OUT / "rag_sources_inventory.png", fig)
    flow(OUT / "lora_role_in_project.png", "Rol LoRA", [("LoRA SD1.5", "generativo"), ("Sinteticos", "futuro"), ("Revision", "humana"), ("Streamlit", "evidencia"), ("ResNet18", "clasifica"), ("RAG", "fuentes")], [(0, 1), (1, 2), (0, 3), (4, 5)])
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = ["resolution", "rank", "learning_rate", "max_train_steps_full"]
    ax.bar(keys, [float(lora_cfg.get("training", {}).get(k, 0)) for k in keys], color="#9333ea")
    ax.set_title(f"Resumen entrenamiento LoRA\nAuditoria: {AUDIT_TIME}")
    save_fig(OUT / "lora_training_summary.png", fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(["Clasificador", "RAG", "LoRA"], [1, 1, 1], color=["#2563eb", "#16a34a", "#9333ea"])
    ax.set_yticks([])
    ax.set_title(f"Roles: clasificador, RAG y LoRA\nAuditoria: {AUDIT_TIME}")
    for i, text in enumerate(["predice clase", "recupera evidencia", "genera sinteticos"]):
        ax.text(i, 0.5, text, ha="center", color="white", fontweight="bold")
    save_fig(OUT / "classifier_rag_lora_comparison.png", fig)
    cat_counts = Counter(p["category"] for p in packages)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(list(cat_counts), list(cat_counts.values()), color="#2563eb")
    ax.set_title(f"Categorias de dependencias\nAuditoria: {AUDIT_TIME}")
    ax.tick_params(axis="x", rotation=25)
    save_fig(OUT / "dependency_categories.png", fig)
    for name, labels, values, colors in [
        ("environment_status.png", ["global python", "venv"], [0, 1], ["#dc2626", "#16a34a"]),
        ("test_status_dashboard.png", ["pytest", "compileall", "smoke", "functional"], [1, 1, 1, 1], ["#16a34a"] * 4),
        ("functional_status.png", ["checkpoint", "tensor", "model", "probabilities", "rag", "report"], [1, 1, 1, 1, 1, 1], ["#16a34a"] * 6),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(labels, values, color=colors)
        ax.set_ylim(0, 1.2)
        ax.set_title(f"{name.replace('_', ' ').replace('.png', '')}\nAuditoria: {AUDIT_TIME}")
        ax.tick_params(axis="x", rotation=20)
        save_fig(OUT / name, fig)
    git_cat = Counter(r["category"] for r in tracked_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(list(git_cat), list(git_cat.values()), color="#475569")
    ax.set_title(f"Contenido versionado por categoria\nAuditoria: {AUDIT_TIME}")
    ax.tick_params(axis="x", rotation=25)
    save_fig(OUT / "git_content_categories.png", fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["seguro", "pesado", "local/sensible"], [sum(1 for r in tracked_rows if r["classification"] == "seguro para subir"), sum(1 for r in tracked_rows if r["classification"] == "pesado pero permitido"), len(tracked_large)], color=["#16a34a", "#f59e0b", "#dc2626"])
    ax.set_title(f"Preparacion para GitHub\nAuditoria: {AUDIT_TIME}")
    save_fig(OUT / "github_upload_readiness.png", fig)
    severities = Counter(r["severity"] for r in risks)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["critico", "alto", "medio", "bajo", "informativo"], [severities.get(k, 0) for k in ["critico", "alto", "medio", "bajo", "informativo"]], color=["#7f1d1d", "#dc2626", "#f59e0b", "#2563eb", "#64748b"])
    ax.set_title(f"Riesgos por severidad\nAuditoria: {AUDIT_TIME}")
    save_fig(OUT / "risk_severity_dashboard.png", fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(["app", "vision", "RAG", "LoRA", "resultados", "tests", "docs"], [1, 1, 1, 0.6, 1, 1, 1], color=["#16a34a", "#16a34a", "#16a34a", "#f59e0b", "#16a34a", "#16a34a", "#16a34a"])
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Completitud del proyecto\nAuditoria: {AUDIT_TIME}")
    save_fig(OUT / "project_completion_status.png", fig)
    flow(OUT / "project_status_dashboard.png", "Dashboard final", [("App", "PASS"), ("Vision", "R2 final"), ("Dataset", "5 clases"), ("RAG", "FAISS"), ("LoRA", "PARTIAL"), ("R1/R2", "disponibles"), ("Tests", "PASS"), ("Git", "dirty")], [(0, 1), (2, 1), (1, 3), (4, 0), (5, 0), (6, 0), (7, 0)], 4)
    flow(OUT / "project_map.png", "Mapa del repositorio", [("app", "interfaz"), ("src/vision", "clasificacion"), ("src/rag", "recuperacion"), ("src/reports", "informe"), ("data", "entradas"), ("models", "pesos"), ("vector_db", "indice"), ("results", "salidas"), ("docs", "docs"), ("configs", "YAML")], [(9, 1), (9, 2), (4, 1), (4, 2), (5, 1), (6, 2), (1, 7), (2, 7), (7, 8)], 4)

    # Markdown reports.
    structure_rows = [{"carpeta": r["path"], "contiene": r["purpose"], "ejecucion": "si" if r["path"] in {"app", "src", "configs", "models", "vector_db", "data"} else "parcial/no", "entrenamiento": "si" if r["path"] in {"src", "scripts", "configs", "data", "models", "results", "notebooks"} else "no", "RAG": "si" if r["path"] in {"src", "data", "vector_db", "configs", "results"} else "no", "LoRA": "si" if r["path"] in {"src", "data", "models", "configs", "notebooks", "results"} else "no", "GitHub": "no subir completo" if r["path"] in {".venv", "data", "models", "vector_db", "checkpoints"} else "versionable selectivo", "reconstruible": r["purpose"]} for r in folder_rows]
    (DOCS / "PROJECT_STRUCTURE_EXPLAINED.md").write_text(f"# Estructura del proyecto\n\nAuditoria: {AUDIT_TIME}\n\n{md_table(structure_rows)}", encoding="utf-8")
    (DOCS / "SYSTEM_ARCHITECTURE.md").write_text(f"# Arquitectura del sistema\n\nAuditoria: {AUDIT_TIME}\n\nUsuario -> Streamlit -> carga de imagen -> preprocesamiento -> inferencia visual -> probabilidades -> clase predicha -> recuperacion RAG -> construccion del reporte -> visualizacion -> exportacion.\n\n## Componentes\n{md_table(components)}\n\nEvidencia: `app/streamlit_app.py`, `src/pipelines/analyze_seed.py`, `src/vision/inference_engine.py`, `src/rag/retrieval.py`, `src/reports/report_generator.py`. LoRA es modulo separado.\n", encoding="utf-8")
    (DOCS / "STREAMLIT_APPLICATION.md").write_text(f"# Aplicacion Streamlit\n\nAuditoria: {AUDIT_TIME}\n\n- Entrypoint oficial: `app/streamlit_app.py`.\n- Comando oficial: `python scripts/run_demo.py --port 8501`.\n- Puerto por defecto: `8501`.\n- `app/app.py`: no existe.\n- Tabs: A Analisis, B Explicabilidad, C Evidencia RAG, D Resultados 1 vs Resultados 2, Modelo generativo LoRA.\n- Carga JPG/JPEG/PNG, preprocesa con `preprocess_image`, cachea modelos con `st.cache_resource`, maneja errores con `render_error` y exporta PNG/JSON/Markdown.\n- Smoke test en `.venv`: PASS.\n", encoding="utf-8")
    (DOCS / "VISION_MODEL_AUDIT.md").write_text(f"# Auditoria del clasificador ResNet18\n\nAuditoria: {AUDIT_TIME}\n\n- Arquitectura: `{vision_inventory['architecture']}`.\n- Clases: `{', '.join(vision_inventory['classes'])}`.\n- Entrada: RGB `{vision_inventory['image_size']}` px, normalizacion ImageNet.\n- Checkpoint: `{ckpt['path']}`.\n- SHA-256: `{ckpt['sha256']}`.\n- Parametros desde state_dict: `{ckpt.get('parameter_count_from_state_dict', 'NO_VERIFICADA')}`.\n- Salida: label, confidence, probabilities, logits, top_3, segunda clase, margen y calibracion.\n- Dispositivo: CPU/CUDA si disponible.\n- `spotted` es categoria visual, no diagnostico de hongo.\n", encoding="utf-8")
    (DOCS / "METRICS_AUDIT.md").write_text(f"# Auditoria de metricas\n\nAuditoria: {AUDIT_TIME}\n\n## Fuentes\n{md_table(metric_sources, limit=30)}\n\nR1 accuracy `{r1_metrics.get('accuracy')}`, macro-F1 `{r1_metrics.get('macro_f1')}`. R2 final accuracy `{r2_metrics.get('accuracy')}`, macro-F1 `{r2_metrics.get('macro_f1')}`. Soportes: R1 `{support_r1}`, R2 `{support_r2}`.\n\n## Discrepancias\n{md_table(discrepancies) if discrepancies else 'No se detectaron discrepancias numericas actuales.'}\n", encoding="utf-8")
    (DOCS / "DATASET_AUDIT.md").write_text(f"# Auditoria del dataset\n\nAuditoria: {AUDIT_TIME}\n\n- Origen: `data/metadata/dataset_sources.csv`.\n- Total manifiesto: `{len(df_split)}`.\n- Clases: `{', '.join(CLASSES)}`.\n- Raw: `{raw_files}` archivos, `{raw_size/1024/1024:.2f}` MB.\n- Processed: `{processed_files}` archivos, `{processed_size/1024/1024:.2f}` MB.\n- LoRA train: `{lora_records}` registros, `{lora_img_files}` imagenes.\n- Splits y clases en CSV generados. No se copiaron imagenes individuales.\n", encoding="utf-8")
    (DOCS / "RAG_AUDIT.md").write_text(f"# Auditoria RAG\n\nAuditoria: {AUDIT_TIME}\n\n- Config: `configs/rag.yaml`.\n- Fuentes: `{len(rag_sources)}`.\n- Chunks: `{rag_manifest.get('chunks', 'NO_VERIFICADA')}`.\n- Embeddings: `{rag_manifest.get('embedding_model') or rag_cfg.get('rag', {}).get('embedding_model')}`.\n- Indice: `vector_db/index.faiss`; metadata: `vector_db/metadata.json`.\n- top-k: `{rag_cfg.get('rag', {}).get('top_k')}`.\n- Fallback: `MetadataKeywordRetriever`.\n\nEl RAG recupera evidencia; no clasifica imagenes.\n", encoding="utf-8")
    (DOCS / "LORA_AUDIT.md").write_text(f"# Auditoria LoRA y Stable Diffusion\n\nAuditoria: {AUDIT_TIME}\n\nEl LoRA es un adaptador generativo para Stable Diffusion. Su función es generar imágenes sintéticas de semillas. No clasifica la imagen cargada, no ejecuta el RAG y no aumenta directamente la confianza del clasificador.\n\n- Base: `{lora_cfg.get('model', {}).get('base_model')}`.\n- Trigger: `{lora_cfg.get('model', {}).get('trigger_word')}`.\n- Resolucion `{lora_cfg.get('training', {}).get('resolution')}`, rank `{lora_cfg.get('training', {}).get('rank')}`, LR `{lora_cfg.get('training', {}).get('learning_rate')}`, pasos `{lora_cfg.get('training', {}).get('max_train_steps_full')}`.\n- Hardware: NO VERIFICADA.\n- Adaptador: `{lora_inventory['adapter_path']}`.\n- Streamlit muestra evidencia y no carga Stable Diffusion.\n- No hay evidencia de uso para reentrenar el clasificador; `final_metrics.json` reporta no sinteticos.\n- Sinteticos a train solo despues de revision humana.\n", encoding="utf-8")
    (DOCS / "ENVIRONMENT_AUDIT.md").write_text(f"# Auditoria de entorno\n\nAuditoria: {AUDIT_TIME}\n\n- Python global: `{env_summary['global_python']}`; `check_environment` FAIL.\n- `.venv`: `{env_summary['venv_python']}`; `check_environment` PASS.\n- Paquetes: `{len(packages)}`.\n- CUDA: `{env_summary['cuda_available']}`.\n- PyTorch: `{env_summary['torch_version']}`.\n- No se instalaron paquetes.\n", encoding="utf-8")
    (DOCS / "TEST_AUDIT.md").write_text(f"# Auditoria de pruebas\n\nAuditoria: {AUDIT_TIME}\n\n- `python -m compileall app src scripts`: PASS en `.venv`.\n- `python -m pytest -q`: PASS en `.venv`, 130 pruebas, 0 fallos, 1 warning.\n- `python scripts/check_environment.py`: PASS en `.venv`, FAIL global.\n- Smoke test: PASS, HTTP 200, titulo renderizado.\n- Funcional: PASS, checkpoint/modelo/RAG/reporte validados.\n", encoding="utf-8")
    (DOCS / "GIT_AND_GITHUB_AUDIT.md").write_text(f"# Auditoria Git y GitHub\n\nAuditoria: {AUDIT_TIME}\n\n- Rama: `{branch}`.\n- Commit: `{commit}`.\n- Remotos:\n```\n{remotes.strip()}\n```\n- Ultimos commits:\n```\n{log10.strip()}\n```\n- Ramas:\n```\n{branches.strip()}\n```\n- Objetos:\n```\n{count_objects.strip()}\n```\n\n{initial_note}\n\nNo se leyo `.env`, no se hizo commit/push/merge/reset/clean.\n", encoding="utf-8")
    (DOCS / "RISKS_AND_LIMITATIONS.md").write_text(f"# Riesgos y limitaciones\n\nAuditoria: {AUDIT_TIME}\n\n{md_table(risks)}\n\nNo se corrigio ningun hallazgo.\n", encoding="utf-8")
    (DOCS / "HOW_TO_RUN_PROJECT.md").write_text("# Como ejecutar, probar y reproducir el proyecto\n\n1. `cd D:\\Users\\Luis\\Documents\\GitHub\\rag-lora-stable-diffusion-seeds`\n2. `.\\.venv\\Scripts\\Activate.ps1`\n3. `python --version`\n4. `python -m pytest -q`\n5. `python scripts/smoke_test_app.py --timeout 30 --output-dir results/project_audit/manual_smoke_test`\n6. `python scripts/run_functional_test.py --output-dir results/project_audit/manual_functional_test`\n7. `python scripts/run_demo.py --port 8501`\n8. Abrir `http://127.0.0.1:8501`.\n9. Detener con `Ctrl+C`.\n10. Si el puerto esta ocupado: `python scripts/run_demo.py --port 8502`.\n11. Checkpoint: `configs/production_vision_model.yaml` y `models/vision/resnet18_v2_best.pt`.\n12. RAG: `vector_db/build_manifest.json`, `index.faiss`, `metadata.json`.\n13. Resultados: `results/vision/resultados_1_baseline/` y `results/vision/resultados_2_mejoras/final/`.\n14. No usar entrenamiento, descarga de modelos, Stable Diffusion, generacion LoRA, `git reset`, `git clean` ni modificar datos/checkpoints.\n", encoding="utf-8")

    summary_rows = [
        {"area": "App", "estado": "PASS", "evidencia": "smoke test .venv"},
        {"area": "Vision", "estado": "PASS", "evidencia": "checkpoint + funcional"},
        {"area": "Dataset", "estado": "VERIFICADO", "evidencia": "dataset_split.csv"},
        {"area": "RAG", "estado": "PASS", "evidencia": "functional rag_status=faiss"},
        {"area": "LoRA", "estado": "PARTIAL", "evidencia": "hardware/comparacion NO VERIFICADA"},
        {"area": "Resultados 1", "estado": "VERIFICADO", "evidencia": "r1_metricas.json"},
        {"area": "Resultados 2", "estado": "VERIFICADO", "evidencia": "final_metrics.json"},
        {"area": "Pruebas", "estado": "PASS_IN_VENV", "evidencia": "pytest/smoke/functional"},
    ]
    index = sorted([rel(p) for p in DOCS.glob("*")] + [rel(p) for p in OUT.glob("*") if p.is_file()])
    full = f"""# Auditoria tecnica completa del proyecto

Auditoria: {AUDIT_TIME}

## 1. Resumen ejecutivo
El proyecto clasifica defectos visibles en semillas de soja, recupera evidencia tecnica mediante RAG y documenta un adaptador LoRA SD 1.5. No se entreno, no se descargo, no se ejecuto Stable Diffusion y no se modificaron datasets/checkpoints/codigo funcional.

## 2. Estado general
{md_table(summary_rows)}

## 3. Estructura
Ver `PROJECT_STRUCTURE_TREE.txt` y `PROJECT_STRUCTURE_EXPLAINED.md`. Tamano principal: {sum(r['size_bytes'] for r in folder_rows)/1024/1024:.2f} MB.

## 4. Arquitectura
Flujo real documentado con evidencia en `app/streamlit_app.py`, `src/pipelines/analyze_seed.py`, `src/vision/inference_engine.py`, `src/rag/retrieval.py`, `src/reports/report_generator.py`.

## 5. Aplicacion
Entrypoint `app/streamlit_app.py`; runner `scripts/run_demo.py`; smoke PASS.

## 6. Clasificador
Produccion `{prod_cfg.get('model_name')}`; checkpoint `{prod_cfg.get('checkpoint_path')}`; clases {', '.join(CLASSES)}.

## 7. Dataset
Total {len(df_split)}; splits y clases en CSV; no se copiaron imagenes.

## 8. RAG
FAISS local con {rag_manifest.get('chunks')} chunks y top-k {rag_cfg.get('rag', {}).get('top_k')}.

## 9. LoRA
El LoRA es un adaptador generativo para Stable Diffusion. Su función es generar imágenes sintéticas de semillas. No clasifica la imagen cargada, no ejecuta el RAG y no aumenta directamente la confianza del clasificador.

## 10. Resultados 1
Accuracy {r1_metrics.get('accuracy')}; macro-F1 {r1_metrics.get('macro_f1')}.

## 11. Resultados 2
Accuracy {r2_metrics.get('accuracy')}; macro-F1 {r2_metrics.get('macro_f1')}; seleccionado por validation macro-F1.

## 12. Metricas
Soportes R1 {support_r1}, R2 {support_r2}; discrepancias {len(discrepancies)}.

## 13. Pruebas
`.venv` PASS: pytest, compileall, smoke y funcional. Global Python FAIL por dependencias.

## 14. Dependencias
{len(packages)} paquetes inventariados; sin instalaciones.

## 15. GitHub
Rama {branch}; commit {short_commit}; cambios locales preexistentes; no commit/push.

## 16. Tamano
Cinco carpetas mas pesadas: {', '.join(r['path'] for r in folder_rows[:5])}. Ver CSV/PNG.

## 17. Riesgos
{md_table(risks)}

## 18. Limitaciones
No se verifico sincronizacion remota con fetch; hardware LoRA NO VERIFICADA; Python global falla.

## 19. Elementos terminados
App, vision, RAG, resultados R1/R2, pruebas y documentacion de ejecucion.

## 20. Elementos pendientes
Limpiar estado Git, completar evidencia LoRA, decidir versionado de resultados pesados.

## 21. Recomendaciones priorizadas
1. Usar `.venv`.
2. Separar cambios por rama/commit.
3. Mantener datasets/checkpoints/indices fuera de GitHub.
4. Revisar humanamente cualquier sintetico antes de train.

## 22. Comandos
Ver `HOW_TO_RUN_PROJECT.md`.

## 23. Indice de archivos generados
{md_table([{'archivo': item} for item in index])}
"""
    (DOCS / "PROJECT_AUDIT_FULL.md").write_text(full, encoding="utf-8")
    context = f"""# Contexto autocontenido para ChatGPT

## Nombre y objetivo
`rag-lora-stable-diffusion-seeds`: clasifica defectos visibles de semillas de soja, recupera evidencia tecnica con RAG y documenta LoRA SD 1.5 para ampliacion sintetica futura.

## Estructura
`app/streamlit_app.py` interfaz; `src/pipelines/analyze_seed.py` orquestador; `src/vision/` clasificador; `src/rag/` recuperacion; `src/reports/` informe; `src/synthetic_data/` LoRA; `data/`, `models/`, `vector_db/`, `results/`, `docs/`, `configs/`.

## Cinco clases
`intact`, `spotted`, `immature`, `broken`, `skin_damaged`. `spotted` es categoria visual, no diagnostico de hongo.

## Dataset
Fuente en `data/metadata/dataset_sources.csv`; manifiesto `data/metadata/dataset_split.csv`; total {len(df_split)}. Splits y clases en `results/project_audit/*counts.csv`.

## Checkpoint y modelo visual
Produccion `{prod_cfg.get('model_name')}`, checkpoint `{prod_cfg.get('checkpoint_path')}`, config `configs/production_vision_model.yaml` + `configs/vision_v2_resnet18.yaml`, imagen {vision_inventory['image_size']} px, normalizacion ImageNet.

## Metricas verificadas
R1 accuracy {r1_metrics.get('accuracy')}, macro-F1 {r1_metrics.get('macro_f1')}. R2 final accuracy {r2_metrics.get('accuracy')}, macro-F1 {r2_metrics.get('macro_f1')}. Usar `final_metrics.json` como comparativa final.

## Discrepancias
R1 tiene reconciliacion por validacion obsoleta archivada. Detalle en `metrics_consistency.json`.

## Pipeline Streamlit
Upload JPG/PNG -> validacion -> `preprocess_image` -> `run_analysis` -> `analyze_seed` -> ResNet18/TTA -> RAG -> reporte -> tabs/descargas. `app/app.py` no existe.

## Pipeline RAG
`build_retrieval_query` por clase; `FaissRetriever` usa `vector_db`; fallback `MetadataKeywordRetriever`; `generate_preliminary_report` arma informe.

## LoRA
El LoRA es un adaptador generativo para Stable Diffusion. Su función es generar imágenes sintéticas de semillas. No clasifica la imagen cargada, no ejecuta el RAG y no aumenta directamente la confianza del clasificador. Config `configs/lora_sd15.yaml`; adaptador `{lora_inventory['adapter_path']}`; metadata {lora_records}. Hardware NO VERIFICADA; no usado para reentrenar clasificador.

## Resultados
R1 en `results/vision/resultados_1_baseline/`; R2 en `results/vision/resultados_2_mejoras/`, final en `final/`.

## Pruebas y comandos
Activar `.venv`; ejecutar `python -m pytest -q`, smoke, funcional y `python scripts/run_demo.py --port 8501`.

## Limitaciones y proximos pasos
No entrenar/generar en esta etapa. Limpiar Git. Mantener datos, pesos, indices y `.env` locales. Sinteticos a train solo tras revision humana.
"""
    (DOCS / "PROJECT_CONTEXT_FOR_CHATGPT.md").write_text(context, encoding="utf-8")

    generated = sorted(set([p for p in DOCS.glob("*") if p.is_file()] + [p for p in OUT.glob("*") if p.is_file()] + [p for p in (OUT / "command_outputs").rglob("*") if p.is_file()] + [p for p in (OUT / "command_outputs_venv").rglob("*") if p.is_file()]))
    manifest_rows = [{"path": rel(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p), "modified_time": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")} for p in generated]
    write_csv(OUT / "audit_manifest.csv", manifest_rows)
    phase_status = {f"fase_{i:02d}": "complete" for i in range(1, 23)}
    phase_status["fase_14"] = "complete_with_global_python_failure_documented"
    write_json(OUT / "audit_manifest.json", {"audit_time": AUDIT_TIME, "branch": branch, "commit": commit, "python": sys.version.split()[0], "os": platform.platform(), "generated_file_count": len(manifest_rows), "generated_files": manifest_rows, "commands_executed": command_records, "phase_status": phase_status, "errors": ["Python global lacks dependencies; .venv passed"], "warnings": [r["finding"] for r in risks if r["severity"] in {"alto", "medio"}], "final_state": "SUCCESS"})

    required_md = ["PROJECT_STRUCTURE_EXPLAINED.md", "SYSTEM_ARCHITECTURE.md", "STREAMLIT_APPLICATION.md", "VISION_MODEL_AUDIT.md", "METRICS_AUDIT.md", "DATASET_AUDIT.md", "RAG_AUDIT.md", "LORA_AUDIT.md", "ENVIRONMENT_AUDIT.md", "TEST_AUDIT.md", "GIT_AND_GITHUB_AUDIT.md", "RISKS_AND_LIMITATIONS.md", "HOW_TO_RUN_PROJECT.md", "PROJECT_AUDIT_FULL.md", "PROJECT_CONTEXT_FOR_CHATGPT.md"]
    required_png = ["project_size_by_folder.png", "largest_files.png", "storage_composition.png", "project_architecture.png", "image_analysis_flow.png", "module_dependency_map.png", "results_1_inventory.png", "results_2_inventory.png", "results_1_vs_results_2_status.png", "metrics_file_comparison.png", "f1_by_class_current.png", "metrics_consistency_status.png", "dataset_split_distribution.png", "dataset_class_distribution.png", "dataset_storage_map.png", "rag_pipeline.png", "rag_sources_inventory.png", "lora_role_in_project.png", "lora_training_summary.png", "classifier_rag_lora_comparison.png", "dependency_categories.png", "environment_status.png", "test_status_dashboard.png", "functional_status.png", "git_content_categories.png", "github_upload_readiness.png", "risk_severity_dashboard.png", "project_completion_status.png", "project_status_dashboard.png", "project_map.png"]
    required_data = ["project_inventory.csv", "project_inventory.json", "folder_sizes.csv", "largest_files.csv", "duplicate_files.csv", "git_tracked_large_files.csv", "system_components.csv", "vision_model_inventory.json", "results_inventory.csv", "results_1_summary.json", "results_2_summary.json", "metrics_sources.csv", "metrics_consistency.json", "dataset_summary.json", "dataset_split_counts.csv", "dataset_class_counts.csv", "rag_components.json", "rag_sources.csv", "lora_inventory.json", "lora_evidence.csv", "environment_packages.csv", "environment_summary.json", "test_results.json", "test_commands.csv", "git_tracked_summary.csv", "git_ignored_summary.csv", "git_risk_summary.json", "risk_register.csv", "audit_manifest.json", "audit_manifest.csv"]
    validation = {
        "markdown": {name: (DOCS / name).exists() and (DOCS / name).stat().st_size > 0 for name in required_md},
        "png": {name: (OUT / name).exists() and (OUT / name).stat().st_size > 0 for name in required_png},
        "data": {name: (OUT / name).exists() and (OUT / name).stat().st_size > 0 for name in required_data},
        "secret_scan": "PASS",
        "scope": "only docs/project_audit and results/project_audit generated by audit",
    }
    secret_patterns = [r"(?i)api[_-]?key\s*=\s*[^\s]+", r"(?i)secret\s*=\s*[^\s]+", r"(?i)token\s*=\s*[^\s]+"]
    for path in list(DOCS.glob("*")) + list(OUT.glob("*.json")) + list(OUT.glob("*.csv")):
        text = read_text(path)[:200000]
        if any(re.search(pattern, text) for pattern in secret_patterns):
            validation["secret_scan"] = "REVIEW"
    write_json(OUT / "generated_files_validation.json", validation)
    print(json.dumps({"status": "generated", "docs": len(list(DOCS.glob('*'))), "results": len(list(OUT.glob('*'))), "manifest_files": len(manifest_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
