"""Generate fold TSVs using the current project's fold strategies.

Not used by the article-reproduction pipeline. Retained to generate
new fold assignments if RepoRT dataset is refreshed.

This script operates on the processed TSV consumed by the current
project model. The resulting ``fold_*.tsv`` files can then be mapped to
GraphormerRT rows with ``build_shared_fold_assets.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _subset_sizes(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    return (
        df.groupby(column_name, sort=False)
        .size()
        .reset_index(name="size")
        .sort_values(by="size", ascending=False)
        .reset_index(drop=True)
    )


def _group_kfold(df: pd.DataFrame, column_name: str, n_splits: int) -> list[pd.DataFrame]:
    fold_groups = [[] for _ in range(n_splits)]
    fold_sizes = [0 for _ in range(n_splits)]

    for _, row in _subset_sizes(df, column_name).iterrows():
        fold_index = int(np.argmin(fold_sizes))
        fold_groups[fold_index].append(row[column_name])
        fold_sizes[fold_index] += int(row["size"])

    folds = []
    for groups in fold_groups:
        fold_df = df[df[column_name].isin(groups)].copy()
        folds.append(fold_df)
    return folds


def _random_within_cc_kfold(df: pd.DataFrame, n_splits: int, seed: int, cc_col: str) -> list[pd.DataFrame]:
    fold_parts = [[] for _ in range(n_splits)]
    for _, cc_df in df.groupby(cc_col, sort=False):
        splitter = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        for fold_index, (_, held_out_index) in enumerate(splitter.split(cc_df)):
            fold_parts[fold_index].append(cc_df.iloc[held_out_index])
    return [pd.concat(parts, ignore_index=True) for parts in fold_parts]


def _write_report(folds: list[pd.DataFrame], output_dir: Path, strategy: str, group_column: str | None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fold_index, fold_df in enumerate(folds):
        fold_df.to_csv(output_dir / f"fold_{fold_index}.tsv", sep="\t", index=False)
        row = {
            "fold": fold_index,
            "molecule_count": int(len(fold_df)),
        }
        if group_column and group_column in fold_df.columns:
            row[f"{group_column}_count"] = int(fold_df[group_column].nunique())
        if "cc_id" in fold_df.columns:
            row["cc_count"] = int(fold_df["cc_id"].nunique())
        if "ms_smiles" in fold_df.columns:
            row["scaffold_count"] = int(fold_df["ms_smiles"].nunique())
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "fold_summary.tsv", sep="\t", index=False)

    summary = {
        "strategy": strategy,
        "group_column": group_column,
        "n_splits": len(folds),
        "n_rows": int(sum(len(fold_df) for fold_df in folds)),
        "folds": rows,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    project_root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=project_root / "data" / "RepoRT_RP" / "processed_data" / "no_SMRT" / "complete_processed_data.tsv",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=project_root / "data" / "benchmarks" / "shared_folds" / "chromagrt_cc_kfold_folds",
        type=Path,
    )
    parser.add_argument("--n-splits", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--strategy",
        choices=["cc", "scaffold", "random_within_cc"],
        default="cc",
        help=(
            "cc and scaffold reproduce the greedy group balancing used by the "
            "current project. random_within_cc reproduces the random k-fold "
            "strategy that splits each cc_id internally."
        ),
    )
    parser.add_argument("--cc-col", default="cc_id")
    parser.add_argument("--scaffold-col", default="ms_smiles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, sep="\t")

    if args.strategy == "cc":
        folds = _group_kfold(df, args.cc_col, args.n_splits)
        group_column = args.cc_col
    elif args.strategy == "scaffold":
        folds = _group_kfold(df, args.scaffold_col, args.n_splits)
        group_column = args.scaffold_col
    else:
        folds = _random_within_cc_kfold(df, args.n_splits, args.seed, args.cc_col)
        group_column = None

    summary = _write_report(folds, args.output_dir, args.strategy, group_column)
    print(f"fold_dir: {args.output_dir}")
    print(f"rows: {summary['n_rows']}")
    print(f"folds: {summary['n_splits']}")


if __name__ == "__main__":
    main()
