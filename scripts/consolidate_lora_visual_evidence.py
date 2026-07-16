"""Consolidate visual evidence for the trained LoRA section in Streamlit.

This script does not train, generate images, load Stable Diffusion, or read
safetensors tensors. It only reads existing local evidence and writes display
artifacts under results/vision/resultados_2_mejoras/10_lora_generativo/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.synthetic_data.lora_evidence import write_visual_evidence_bundle


def main() -> None:
    """CLI entrypoint."""

    evidence = write_visual_evidence_bundle()
    summary = {
        "status": evidence["status"],
        "evidence_found": evidence["evidence_found"],
        "evidence_missing": evidence["evidence_missing"],
        "generated_pngs": evidence["generated_pngs"],
        "output_dir": "results/vision/resultados_2_mejoras/10_lora_generativo",
        "stable_diffusion_loaded": evidence["stable_diffusion_loaded"],
        "lora_inference_required": evidence["lora_inference_required"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
