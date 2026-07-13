from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.analyze_seed import analyze_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for seed analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze a soybean seed image with ResNet18, RAG and a preliminary report."
    )
    parser.add_argument("--image", type=Path, required=True, help="Path to the input seed image.")
    parser.add_argument(
        "--vision-config",
        type=Path,
        default=Path("configs/vision_config.yaml"),
        help="Path to the vision YAML config.",
    )
    parser.add_argument(
        "--rag-config",
        type=Path,
        default=Path("configs/rag.yaml"),
        help="Path to the RAG YAML config.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("vector_db"),
        help="Directory containing index.faiss and metadata.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional ResNet18 checkpoint path. Defaults to the vision config model_dir.",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Number of RAG fragments to retrieve.")
    parser.add_argument(
        "--observation",
        action="append",
        default=None,
        help="Optional visual observation. May be passed more than once.",
    )
    parser.add_argument("--device", default=None, help="Optional torch device, for example cpu or cuda.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the JSON report payload will be written.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    result = analyze_seed(
        image=args.image,
        vision_config_path=args.vision_config,
        rag_config_path=args.rag_config,
        index_dir=args.index,
        checkpoint_path=args.checkpoint,
        observations=args.observation,
        top_k=args.top_k,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
