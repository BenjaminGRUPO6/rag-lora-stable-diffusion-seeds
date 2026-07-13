from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.evaluate_system import (
    N_A,
    build_outputs,
    consolidate_lora,
    consolidate_vision,
    select_demo_cases,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    vision = tmp_path / "vision"
    rag = tmp_path / "rag"
    lora = tmp_path / "lora"
    manifest = tmp_path / "dataset_split.csv"
    write_json(
        vision / "metrics_test.json",
        {
            "accuracy": 0.5,
            "macro_precision": 0.6,
            "macro_recall": 0.7,
            "macro_f1": 0.8,
            "per_class": {
                "intact": {"f1": 0.9},
                "broken": {"f1": 0.4},
            },
        },
    )
    write_json(vision / "run_summary.json", {"classes": ["intact", "broken"], "test_samples": 3})
    write_csv(
        vision / "test_predictions.csv",
        [
            {"image_path": "a.jpg", "true_label": "intact", "predicted_label": "intact"},
            {"image_path": "b.jpg", "true_label": "intact", "predicted_label": "broken"},
            {"image_path": "c.jpg", "true_label": "broken", "predicted_label": "broken"},
        ],
    )
    write_json(
        rag / "metrics.json",
        {
            "query_count": 2,
            "hit_at_1": 0.5,
            "hit_at_3": 1.0,
            "hit_at_5": 1.0,
            "mrr": 0.75,
            "mean_retrieval_latency_ms": 12.0,
            "failed_query_ids_at_5": ["RAG999"],
            "human_review": {"metrics": "pending", "status": "pending"},
        },
    )
    write_csv(rag / "query_results.csv", [{"query_id": "RAG999"}])
    write_json(
        lora / "run_manifest.json",
        {
            "status": "PARTIAL",
            "no_retraining_performed": True,
            "confirmed_parameters": {"rank": {"value": 8, "source": "config"}},
            "dataset": {"metadata_records": 10, "class_distribution": {"intact": 5}},
            "samples_copied": [{"sample": "one.jpg"}],
            "missing_evidence": ["hardware missing"],
        },
    )
    write_json(lora / "training_summary.json", {"status": "PARTIAL"})
    write_csv(lora / "base_vs_lora_manifest.csv", [{"class_name": "intact", "evidence_status": "MISSING"}])
    write_csv(
        manifest,
        [
            {
                "image_id": "intact_1",
                "source_path": "raw/intact/1.jpg",
                "processed_path": "data/processed/validation/intact/1.jpg",
                "label": "intact",
                "split": "validation",
                "sha256": "a",
                "is_synthetic": "False",
                "exclusion_status": "included",
            },
            {
                "image_id": "broken_1",
                "source_path": "raw/broken/1.jpg",
                "processed_path": "data/processed/validation/broken/1.jpg",
                "label": "broken",
                "split": "validation",
                "sha256": "b",
                "is_synthetic": "False",
                "exclusion_status": "included",
            },
        ],
    )
    return vision, rag, lora, manifest


def test_consolidate_vision_builds_confusion_matrix(tmp_path: Path) -> None:
    vision, _, _, _ = make_inputs(tmp_path)

    consolidated, missing = consolidate_vision(vision)

    assert consolidated["accuracy"] == 0.5
    assert consolidated["per_class_f1"] == {"intact": 0.9, "broken": 0.4}
    assert consolidated["confusion_matrix"]["intact"]["intact"] == 1
    assert consolidated["confusion_matrix"]["intact"]["broken"] == 1
    assert missing == []


def test_select_demo_cases_uses_non_test_non_synthetic_split(tmp_path: Path) -> None:
    _, _, _, manifest = make_inputs(tmp_path)

    cases, missing = select_demo_cases(manifest, split="validation", class_names=("intact", "broken"))

    assert [case.class_name for case in cases] == ["intact", "broken"]
    assert all(case.split == "validation" for case in cases)
    assert missing == []


def test_lora_missing_evidence_is_marked_na(tmp_path: Path) -> None:
    _, _, lora, _ = make_inputs(tmp_path)

    consolidated, missing = consolidate_lora(lora)

    assert consolidated["base_vs_lora_evidence"] == N_A
    assert consolidated["visual_evaluation_status"] == N_A
    assert "No base-vs-LoRA comparison evidence is available." in missing


def test_build_outputs_writes_expected_artifacts_with_fake_demo(tmp_path: Path) -> None:
    vision, rag, lora, manifest = make_inputs(tmp_path)
    output = tmp_path / "system"

    def fake_analyzer(case):
        return {
            "prediction": case.class_name,
            "confidence": 0.91,
            "uncertainty_status": "certain",
            "retrieved_sources": [{"document_id": "DOC001"}],
            "retrieval_query": "query",
            "processing_times": {
                "vision_seconds": 0.1,
                "retrieval_seconds": 0.2,
                "report_seconds": 0.03,
                "total_seconds": 0.33,
            },
        }

    final = build_outputs(
        vision_results=vision,
        rag_results=rag,
        lora_results=lora,
        output_dir=output,
        dataset_split=manifest,
        demo_split="validation",
        analyzer=fake_analyzer,
        source_paths={"vision_metrics": vision / "metrics_test.json"},
    )

    assert final["system"]["demo_success_count"] == 2
    assert final["system"]["mean_total_seconds"] == 0.33
    assert (output / "final_metrics.json").exists()
    assert (output / "final_metrics.csv").exists()
    assert (output / "demo_cases.csv").exists()
    assert (output / "latency_report.csv").exists()
    assert (output / "failure_cases.csv").exists()
    assert (output / "final_evaluation_report.md").exists()
    assert "RAG human-review metrics are pending." in final["missing_data"]
    assert any(item["item_id"] == "RAG999" for item in final["failures"])
