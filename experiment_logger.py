"""Small file-based experiment recorder using only the Python standard library."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROUND_FIELDS = (
    "global_round",
    "epoch",
    "round_in_epoch",
    "train_loss",
    "train_accuracy",
    "gradient_norm",
    "round_time_seconds",
    "aggregation_time_seconds",
)

METRIC_FIELDS = (
    "global_round",
    "epoch",
    "split",
    "loss",
    "accuracy",
    "error",
    "elapsed_seconds",
)


def get_git_metadata(repo_dir: str | Path = ".") -> dict[str, Any]:
    """Return the current commit and dirty state without modifying the repository."""

    def run_git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    if commit.returncode != 0 or status.returncode != 0:
        return {"git_commit": None, "git_dirty": None}
    return {
        "git_commit": commit.stdout.strip(),
        "git_dirty": bool(status.stdout.strip()),
    }


class ExperimentRecorder:
    """Write one experiment into an independent directory under ``results/``.

    The schema is aggregation-independent. Mean, Krum, Multi-Krum and future
    attack implementations can share the same scalar metric files and put
    method-specific selection information into ``aggregation.jsonl``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        results_dir: str | Path = "results",
        run_name: str | None = None,
    ) -> None:
        self.started_at = datetime.now().astimezone()
        self._start_time = time.perf_counter()
        self._closed = False
        self._last_global_round = 0

        timestamp = self.started_at.strftime("%Y%m%d_%H%M%S_%f")
        suffix = self._safe_name(run_name or self._default_name(config))
        self.run_id = f"{timestamp}_{suffix}"
        self.run_dir = Path(results_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)

        full_config = {
            "schema_version": 1,
            "run_id": self.run_id,
            "run_name": run_name,
            "started_at": self.started_at.isoformat(),
            **config,
        }
        self._write_json(self.run_dir / "config.json", full_config)

        self._round_file = (self.run_dir / "rounds.csv").open(
            "w", encoding="utf-8", newline="", buffering=1
        )
        self._round_writer = csv.DictWriter(self._round_file, fieldnames=ROUND_FIELDS)
        self._round_writer.writeheader()

        self._metric_file = (self.run_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline="", buffering=1
        )
        self._metric_writer = csv.DictWriter(self._metric_file, fieldnames=METRIC_FIELDS)
        self._metric_writer.writeheader()

        self._aggregation_file = (self.run_dir / "aggregation.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )

    @staticmethod
    def _safe_name(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
        return safe or "experiment"

    @staticmethod
    def _default_name(config: dict[str, Any]) -> str:
        aggregator = str(config.get("aggregator", "experiment"))
        attack = str(config.get("attack", "none"))
        num_byzantine = config.get("num_byzantine", 0)
        seed = config.get("training_seed", 0)
        return f"{aggregator}_{attack}_f{num_byzantine}_seed{seed}"

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start_time

    @property
    def last_global_round(self) -> int:
        return self._last_global_round

    def record_round(
        self,
        *,
        global_round: int,
        epoch: int,
        round_in_epoch: int,
        train_loss: float,
        train_accuracy: float,
        gradient_norm: float,
        round_time_seconds: float,
        aggregation_time_seconds: float,
    ) -> None:
        """Append scalar training measurements for one completed round."""
        self._ensure_open()
        self._last_global_round = global_round
        self._round_writer.writerow(
            {
                "global_round": global_round,
                "epoch": epoch,
                "round_in_epoch": round_in_epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "gradient_norm": gradient_norm,
                "round_time_seconds": round_time_seconds,
                "aggregation_time_seconds": aggregation_time_seconds,
            }
        )
        self._round_file.flush()

    def record_metric(
        self,
        *,
        global_round: int,
        epoch: int,
        split: str,
        loss: float,
        accuracy: float,
    ) -> None:
        """Append a train/test evaluation point."""
        self._ensure_open()
        self._metric_writer.writerow(
            {
                "global_round": global_round,
                "epoch": epoch,
                "split": split,
                "loss": loss,
                "accuracy": accuracy,
                "error": 1.0 - accuracy,
                "elapsed_seconds": self.elapsed_seconds,
            }
        )
        self._metric_file.flush()

    def record_aggregation(
        self,
        *,
        global_round: int,
        aggregator: str,
        selected_worker_ids: Iterable[int],
        byzantine_worker_ids: Iterable[int] = (),
        selected_scores: Iterable[float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append aggregation-specific diagnostics for one round."""
        self._ensure_open()
        selected = list(selected_worker_ids)
        byzantine = set(byzantine_worker_ids)
        event: dict[str, Any] = {
            "global_round": global_round,
            "aggregator": aggregator,
            "selected_worker_ids": selected,
            "byzantine_worker_ids": sorted(byzantine),
            "selected_byzantine": [worker_id in byzantine for worker_id in selected],
        }
        if selected_scores is not None:
            event["selected_scores"] = list(selected_scores)
        if extra:
            event.update(extra)
        self._aggregation_file.write(
            json.dumps(event, ensure_ascii=False, default=str) + "\n"
        )
        self._aggregation_file.flush()

    def finalize(self, *, completed: bool, **summary: Any) -> None:
        """Write the final summary. Safe to call before ``close()``."""
        self._ensure_open()
        finished_at = datetime.now().astimezone()
        self._write_json(
            self.run_dir / "summary.json",
            {
                "completed": completed,
                "finished_at": finished_at.isoformat(),
                "total_time_seconds": self.elapsed_seconds,
                **summary,
            },
        )

    def close(self) -> None:
        if self._closed:
            return
        self._round_file.close()
        self._metric_file.close()
        self._aggregation_file.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ExperimentRecorder is already closed")
