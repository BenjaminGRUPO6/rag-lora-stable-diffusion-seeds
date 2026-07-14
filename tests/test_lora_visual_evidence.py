from __future__ import annotations

from pathlib import Path

import src.synthetic_data.lora_evidence as lora_evidence


def test_lora_visual_evidence_works_without_safetensors(monkeypatch) -> None:
    """Evidence consolidation must not require a local LoRA weight file."""

    monkeypatch.setattr(lora_evidence, "find_safetensors", lambda: [])

    evidence = lora_evidence.build_lora_evidence()

    assert evidence["model"]["adapter"]["exists"] is False
    assert evidence["model"]["adapter"]["status"] == "EVIDENCE_MISSING"
    assert evidence["stable_diffusion_loaded"] is False


def test_lora_visual_evidence_requires_no_lora_inference() -> None:
    """The visual evidence payload records that no LoRA inference is mandatory."""

    evidence = lora_evidence.build_lora_evidence()

    assert evidence["lora_inference_required"] is False
    assert evidence["no_retraining_performed"] is True
    assert evidence["no_mass_generation_performed"] is True
    assert evidence["classification_impact_claim"] == "NOT_CLAIMED"


def test_lora_visual_evidence_does_not_invent_missing_parameters() -> None:
    """Missing hardware and comparison evidence must remain explicit."""

    evidence = lora_evidence.build_lora_evidence()
    training = evidence["training"]

    assert training["hardware"] == {"value": None, "source": "EVIDENCE_MISSING"}
    assert any("hardware" in item for item in evidence["evidence_missing"])
    assert evidence["comparison"]["status"] == "EVIDENCE_MISSING"
    assert all(
        row["evidence_status"] == "EVIDENCE_MISSING"
        for row in evidence["comparison"]["rows"]
    )


def test_lora_visual_evidence_has_no_private_paths() -> None:
    """The consolidated payload must be safe for app display."""

    evidence = lora_evidence.build_lora_evidence()

    assert not lora_evidence.contains_private_path(evidence)


def test_lora_visual_bundle_writes_expected_files(tmp_path: Path, monkeypatch) -> None:
    """The visual bundle writes JSON, model card, CSV and required PNGs."""

    monkeypatch.setattr(lora_evidence, "VISION_LORA_RESULTS_DIR", tmp_path)

    evidence = lora_evidence.write_visual_evidence_bundle(tmp_path)

    assert (tmp_path / "lora_evidence.json").exists()
    assert (tmp_path / "lora_model_card.md").exists()
    assert (tmp_path / "lora_samples_manifest.csv").exists()
    assert (tmp_path / "r2_lora_model_card.png").exists()
    assert (tmp_path / "r2_lora_base_vs_adaptado.png").exists()
    assert (tmp_path / "r2_lora_clases.png").exists()
    assert (tmp_path / "r2_lora_flujo.png").exists()
    assert "El LoRA genera imágenes sintéticas de semillas" in (
        tmp_path / "lora_model_card.md"
    ).read_text(encoding="utf-8")
    assert len(evidence["generated_pngs"]) == 4
