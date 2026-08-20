"""Read recorded experiments and create comparison plots."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunData:
    path: Path
    config: dict[str, Any]
    metrics: list[dict[str, str]]
    rounds: list[dict[str, str]]
    aggregations: list[dict[str, Any]]
    summary: dict[str, Any]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_run(path: Path) -> RunData:
    config_path = path / "config.json"
    if not config_path.exists():
        raise ValueError(f"Missing config.json in {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary_path = path / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    return RunData(
        path=path,
        config=config,
        metrics=read_csv(path / "metrics.csv"),
        rounds=read_csv(path / "rounds.csv"),
        aggregations=read_jsonl(path / "aggregation.jsonl"),
        summary=summary,
    )


def discover_runs(results_dir: Path) -> list[Path]:
    if not results_dir.exists():
        return []
    return sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir() and (path / "config.json").exists()
    )


def run_label(run: RunData, include_seed: bool = True) -> str:
    custom_name = run.config.get("run_name")
    if custom_name and include_seed:
        return str(custom_name)
    aggregator = str(run.config.get("aggregator", "unknown"))
    attack = str(run.config.get("attack", "none"))
    num_byzantine = run.config.get("num_byzantine", 0)
    label = f"{aggregator} | attack={attack} | f={num_byzantine}"
    selection = run.config.get("byzantine_selection")
    if attack != "none" and selection:
        label += f" | selection={selection}"
    if aggregator in ("krum", "multi-krum"):
        label += f" | krum_f={run.config.get('krum_f', 0)}"
    if aggregator == "multi-krum":
        label += f" | m={run.config.get('multi_krum_m', '?')}"
    if include_seed:
        label += f" | seed={run.config.get('training_seed', '?')}"
    return label


def condition_key(run: RunData) -> tuple[Any, ...]:
    config = run.config
    return (
        config.get("aggregator"),
        config.get("krum_f"),
        config.get("multi_krum_m"),
        config.get("attack"),
        config.get("num_byzantine"),
        config.get("byzantine_selection"),
        json.dumps(config.get("attack_parameters", {}), sort_keys=True),
        config.get("num_workers"),
        config.get("batch_size_per_worker"),
        config.get("learning_rate"),
        config.get("optimizer"),
    )


def metric_series(run: RunData, field: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in run.metrics if row.get("split") == "test"]
    return (
        np.asarray([int(row["global_round"]) for row in rows], dtype=int),
        np.asarray([float(row[field]) for row in rows], dtype=float),
    )


def round_series(run: RunData, field: str, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([int(row["global_round"]) for row in run.rounds], dtype=int),
        np.asarray([float(row[field]) * scale for row in run.rounds], dtype=float),
    )


def selected_byzantine_series(run: RunData) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        row
        for row in run.aggregations
        if isinstance(row.get("selected_byzantine"), list)
        and row["selected_byzantine"]
    ]
    return (
        np.asarray([int(row["global_round"]) for row in rows], dtype=int),
        np.asarray(
            [
                sum(bool(value) for value in row["selected_byzantine"])
                / len(row["selected_byzantine"])
                for row in rows
            ],
            dtype=float,
        ),
    )


def plot_individual_runs(
    axes: list[plt.Axes],
    runs: list[RunData],
    series_functions: list[Callable[[RunData], tuple[np.ndarray, np.ndarray]]],
) -> None:
    for run in runs:
        label = run_label(run)
        for axis, get_series in zip(axes, series_functions):
            x_values, y_values = get_series(run)
            if len(x_values):
                axis.plot(
                    x_values,
                    y_values,
                    label=label,
                    linewidth=1.8,
                    marker="o",
                    markersize=2.5,
                )


def plot_seed_groups(
    axes: list[plt.Axes],
    runs: list[RunData],
    series_functions: list[Callable[[RunData], tuple[np.ndarray, np.ndarray]]],
) -> None:
    groups: dict[tuple[Any, ...], list[RunData]] = {}
    for run in runs:
        groups.setdefault(condition_key(run), []).append(run)

    for group_runs in groups.values():
        label = run_label(group_runs[0], include_seed=False)
        for axis, get_series in zip(axes, series_functions):
            values_by_round: dict[int, list[float]] = {}
            for run in group_runs:
                x_values, y_values = get_series(run)
                for x_value, y_value in zip(x_values, y_values):
                    values_by_round.setdefault(int(x_value), []).append(float(y_value))
            if not values_by_round:
                continue
            x_values = np.asarray(sorted(values_by_round), dtype=int)
            means = np.asarray(
                [np.mean(values_by_round[x]) for x in x_values], dtype=float
            )
            standard_deviations = np.asarray(
                [np.std(values_by_round[x]) for x in x_values], dtype=float
            )
            line = axis.plot(
                x_values,
                means,
                label=label,
                linewidth=2.0,
                marker="o",
                markersize=2.5,
            )[0]
            if len(group_runs) > 1:
                axis.fill_between(
                    x_values,
                    means - standard_deviations,
                    means + standard_deviations,
                    color=line.get_color(),
                    alpha=0.18,
                )


def print_summary(runs: list[RunData]) -> None:
    print(f"Loaded {len(runs)} run(s):")
    for run in runs:
        completed = run.summary.get("completed", "unknown")
        accuracy = run.summary.get("final_test_accuracy")
        accuracy_text = "N/A" if accuracy is None else f"{float(accuracy) * 100:.2f}%"
        print(
            f"- {run_label(run)} | completed={completed} "
            f"| final_test_accuracy={accuracy_text} | {run.path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot comparisons from experiment result directories"
    )
    parser.add_argument(
        "runs",
        nargs="*",
        type=Path,
        help="specific run directories; defaults to every run under --results-dir",
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/comparison.png"),
    )
    parser.add_argument(
        "--aggregate-seeds",
        action="store_true",
        help="group equal configurations and plot mean plus/minus one standard deviation",
    )
    parser.add_argument(
        "--title",
        default="Mean / Krum / Multi-Krum Experiment Comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_paths = args.runs or discover_runs(args.results_dir)
    if not run_paths:
        raise SystemExit(
            f"No experiment runs found under {args.results_dir}. "
            "Run training first or pass run directories explicitly."
        )

    runs = [load_run(path) for path in run_paths]
    print_summary(runs)

    figure, axes_grid = plt.subplots(2, 3, figsize=(18, 9), sharex=False)
    axes = list(axes_grid.flat)
    series_functions: list[Callable[[RunData], tuple[np.ndarray, np.ndarray]]] = [
        lambda run: metric_series(run, "error"),
        lambda run: metric_series(run, "loss"),
        lambda run: round_series(run, "train_loss"),
        lambda run: round_series(run, "gradient_norm"),
        lambda run: round_series(run, "aggregation_time_seconds", scale=1000.0),
        selected_byzantine_series,
    ]

    if args.aggregate_seeds:
        plot_seed_groups(axes, runs, series_functions)
    else:
        plot_individual_runs(axes, runs, series_functions)

    titles = (
        "Test error vs global round",
        "Test loss vs global round",
        "Round train loss",
        "Aggregated gradient norm",
        "Aggregation time",
        "Selected Byzantine fraction",
    )
    y_labels = ("Error", "Loss", "Loss", "L2 norm", "Milliseconds", "Fraction")
    for axis, title, y_label in zip(axes, titles, y_labels):
        axis.set_title(title)
        axis.set_xlabel("Global round")
        axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend(fontsize=8)

    figure.suptitle(args.title)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved comparison figure to: {args.output}")


if __name__ == "__main__":
    main()
