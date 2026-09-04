"""Summarize k-fold evaluation metrics written by GraphormerRT or ChromaGRT."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _read_graphormer_rt_metrics(results_root: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(results_root.glob("fold_*/eval_test/metrics.json")):
        fold_name = metrics_path.parents[1].name
        with metrics_path.open() as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "fold": fold_name,
                "MAE": float(metrics["MAE"]),
                "RMSE": float(metrics["RMSE"]),
                "MRE": float(metrics["MRE"]),
                "n_samples": int(metrics["n_samples"]),
                "metrics_path": str(metrics_path),
            }
        )
    return pd.DataFrame(rows)


def _read_chromagrt_metrics(results_root: Path) -> pd.DataFrame:
    rows = []
    metric_paths = {
        *results_root.glob("fold_*/metrics.txt"),
        *results_root.glob("fold_*/*/metrics.txt"),
    }
    for metrics_path in sorted(metric_paths):
        fold_ancestors = [parent for parent in metrics_path.parents if parent.name.startswith("fold_")]
        fold_name = fold_ancestors[0].name if fold_ancestors else metrics_path.parent.name
        values = {}
        with metrics_path.open() as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, value = line.strip().split(":", 1)
                value = value.strip().split()[0]
                if key in {"MAE", "RMSE", "MRE"}:
                    values[key] = float(value)
        test_path = metrics_path.parent / "Results.tsv"
        n_samples = len(pd.read_csv(test_path, sep="\t")) if test_path.exists() else 0
        if values:
            rows.append({"fold": fold_name, **values, "n_samples": n_samples, "metrics_path": str(metrics_path)})
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("No fold metrics found")
    weights = df["n_samples"].astype(float)
    total = float(weights.sum())
    weighted_means = {
        metric: float((df[metric] * weights).sum() / total)
        for metric in ("MAE", "RMSE", "MRE")
    }
    weighted_stds = {
        metric: float(math.sqrt((((df[metric] - weighted_means[metric]) ** 2) * weights).sum() / total))
        for metric in weighted_means
    }
    return {
        "n_folds": int(len(df)),
        "n_samples": int(weights.sum()),
        "MAE_weighted": weighted_means["MAE"],
        "RMSE_global": float(math.sqrt(((df["RMSE"] ** 2) * weights).sum() / total)),
        "MRE_weighted": weighted_means["MRE"],
        "MAE_fold_std_weighted": weighted_stds["MAE"],
        "RMSE_fold_std_weighted": weighted_stds["RMSE"],
        "MRE_fold_std_weighted": weighted_stds["MRE"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--kind", choices=["graphormer_rt", "chromagrt"], default="graphormer_rt")
    parser.add_argument("--output", default=None, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _read_graphormer_rt_metrics(args.results_root) if args.kind == "graphormer_rt" else _read_chromagrt_metrics(args.results_root)
    summary = _aggregate(df)
    output = args.output or args.results_root / "kfold_summary.json"
    df.to_csv(args.results_root / "kfold_fold_metrics.tsv", sep="\t", index=False)
    with output.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
