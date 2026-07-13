from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VISION_RESULTS_ROOT = Path("results") / "vision"
BASELINE_EXPERIMENT_ID = "resultados_1_baseline"
IMPROVEMENTS_EXPERIMENT_ID = "resultados_2_mejoras"
PROTECTED_EXPERIMENT_IDS = frozenset({BASELINE_EXPERIMENT_ID})


class ResultRegistry:
    """Build result paths and register experiment metadata without accidental overwrites."""

    def __init__(
        self,
        results_root: str | Path = VISION_RESULTS_ROOT,
        protected_experiment_ids: Iterable[str] = PROTECTED_EXPERIMENT_IDS,
    ) -> None:
        self.results_root = Path(results_root)
        self.protected_experiment_ids = frozenset(protected_experiment_ids)

    def experiment_dir(self, experiment_id: str) -> Path:
        """Return the root directory for an experiment id."""
        self._validate_path_part(experiment_id, field_name="experiment_id")
        return self.results_root / experiment_id

    def stage_dir(self, experiment_id: str, stage_id: str | None = None) -> Path:
        """Return the experiment directory or a nested stage directory."""
        root = self.experiment_dir(experiment_id)
        if stage_id is None:
            return root
        self._validate_path_part(stage_id, field_name="stage_id")
        return root / stage_id

    def artifact_path(
        self,
        experiment_id: str,
        relative_path: str | Path,
        *,
        allow_protected: bool = False,
        overwrite: bool = False,
        create_parent: bool = True,
    ) -> Path:
        """Build a writable artifact path and refuse protected or existing targets by default."""
        if experiment_id in self.protected_experiment_ids and not allow_protected:
            raise PermissionError(f"Experiment is protected from writes: {experiment_id}")

        base = self.experiment_dir(experiment_id)
        target = self._safe_join(base, relative_path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {target}")
        if create_parent:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def register_experiment(
        self,
        experiment_id: str,
        *,
        config: dict[str, Any],
        seed: int,
        metadata_filename: str = "experiment_registry.json",
        commit: str | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write metadata for an experiment and return the registry path."""
        path = self.artifact_path(
            experiment_id=experiment_id,
            relative_path=metadata_filename,
            overwrite=overwrite,
        )
        payload = {
            "experiment_id": experiment_id,
            "created_at_utc": utc_now_iso(),
            "commit": commit if commit is not None else current_git_commit(),
            "config": config,
            "seed": int(seed),
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def _safe_join(self, base: Path, relative_path: str | Path) -> Path:
        candidate_relative = Path(relative_path)
        if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
            raise ValueError(f"Unsafe relative path: {relative_path}")
        candidate = base / candidate_relative
        resolved_base = base.resolve()
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(resolved_base)
        except ValueError as exc:
            raise ValueError(f"Path escapes experiment directory: {relative_path}") from exc
        return candidate

    @staticmethod
    def _validate_path_part(value: str, *, field_name: str) -> None:
        path = Path(value)
        if not value or path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
            raise ValueError(f"Invalid {field_name}: {value}")


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_git_commit() -> str | None:
    """Return the current Git commit hash when available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None
