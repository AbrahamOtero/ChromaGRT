"""Build GraphormerRT assets from the current project's full folds.

Not used by the article-reproduction pipeline. Retained to generate
new fold assignments if RepoRT dataset is refreshed.

Unlike ``build_shared_fold_assets.py``, this script does not intersect rows with
an existing GraphormerRT CSV. It creates a new GraphormerRT-compatible CSV from the
current project's fold TSVs and builds a matching ``RP_metadata.pickle`` for the
methods present in those folds.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import pandas as pd

from src.benchmarks.shared_folds.asset_generation import (
    _build_metadata,
    _method_from_molecule_id,
    _write_chromagrt_splits,
    _write_graphormer_rt_csv,
    _write_graphormer_rt_manifests,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _fold_index(path: Path) -> int:
    match = re.search(r"fold_(\d+)\.tsv$", path.name)
    if not match:
        raise ValueError(f"Could not parse fold index from {path}")
    return int(match.group(1))


def _fold_paths(fold_dir: Path) -> list[Path]:
    paths = [path for path in fold_dir.glob("fold_*.tsv") if re.search(r"fold_\d+\.tsv$", path.name)]
    if not paths:
        raise FileNotFoundError(f"No fold_*.tsv files found in {fold_dir}")
    return sorted(paths, key=_fold_index)


def _read_fold_rows(fold_dir: Path, molecule_id_col: str, smiles_col: str, rt_col: str) -> tuple[pd.DataFrame, int]:
    records = []
    paths = _fold_paths(fold_dir)
    for fold_path in paths:
        fold_index = _fold_index(fold_path)
        fold_df = pd.read_csv(fold_path, sep="\t", dtype=str).fillna("")
        for source_row_index, row in fold_df.iterrows():
            method_id = _method_from_molecule_id(row[molecule_id_col])
            records.append(
                {
                    **row.to_dict(),
                    "sample_id": f"full_{len(records):06d}",
                    "row_index": len(records),
                    "fold": fold_index,
                    "graphormer_rt_method_id": method_id,
                    "source_fold_file": str(fold_path),
                    "source_row_index": int(source_row_index),
                    "graphormer_rt_smiles": row[smiles_col].replace("Q", "#"),
                    "rt_seconds": float(row[rt_col]),
                    "rt_minutes": float(row[rt_col]) / 60.0,
                }
            )
    return pd.DataFrame(records), len(paths)


def build_assets(args: argparse.Namespace) -> dict:
    rows, n_folds = _read_fold_rows(args.fold_dir, args.molecule_id_col, args.smiles_col, args.rt_col)
    method_ids = set(rows["graphormer_rt_method_id"])
    metadata, metadata_report = _build_metadata(
        method_ids,
        preprocessed_cc=args.preprocessed_cc,
        preprocessed_grad=args.preprocessed_grad,
        column_properties=args.column_properties,
        gradient_time_scale=args.gradient_time_scale,
        missing_column_properties=args.missing_column_properties,
    )
    missing_methods = set(metadata_report.loc[metadata_report["status"] != "ok", "method_id"])
    if missing_methods:
        if args.missing_metadata_policy == "error":
            raise ValueError(f"Missing GraphormerRT metadata for methods: {sorted(missing_methods)[:20]}")
        rows = rows[~rows["graphormer_rt_method_id"].isin(missing_methods)].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = rows.sort_values("row_index").reset_index(drop=True)
    rows["row_index"] = rows.index
    rows["sample_id"] = rows["row_index"].map(lambda value: f"full_{int(value):06d}")

    _write_graphormer_rt_csv(rows, args.output_dir)
    with (args.output_dir / "RP_metadata.pickle").open("wb") as handle:
        pickle.dump(metadata, handle)
    rows.to_csv(args.output_dir / "master_manifest.csv", index=False)
    metadata_report.to_csv(args.output_dir / "metadata_coverage.tsv", sep="\t", index=False)
    graphormer_rt_counts = _write_graphormer_rt_manifests(rows, args.output_dir, n_folds)
    chromagrt_counts = _write_chromagrt_splits(
        rows,
        args.output_dir,
        n_folds,
        include_source_fold_file=True,
    )

    summary = {
        "fold_dir": str(args.fold_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "n_folds": n_folds,
        "n_rows": int(len(rows)),
        "n_methods": int(rows["graphormer_rt_method_id"].nunique()),
        "n_cc_ids": int(rows["cc_id"].nunique()) if "cc_id" in rows.columns else None,
        "n_metadata_methods": int(len(metadata) - 1),
        "n_missing_metadata_methods": int(len(missing_methods)),
        "missing_metadata_policy": args.missing_metadata_policy,
        "gradient_time_scale": args.gradient_time_scale,
        "metadata_encoding": "GraphormerRT-compatible gradient block",
        "graphormer_rt_split_counts": graphormer_rt_counts,
        "chromagrt_split_counts": chromagrt_counts,
        "split_policy": "for run i: test=fold_i, valid=fold_(i+1)%K, train=all remaining folds",
    }
    with (args.output_dir / "mapping_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    pd.DataFrame(
        [{"fold": key, **value} for key, value in sorted(graphormer_rt_counts.items())]
    ).to_csv(args.output_dir / "fold_split_summary.tsv", sep="\t", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    project_root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-dir",
        default=project_root / "data" / "benchmarks" / "shared_folds" / "chromagrt_cc_kfold_folds",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=project_root / "data" / "benchmarks" / "shared_folds" / "report_rp_no_smrt_cc_kfold_chromagrt_data",
        type=Path,
    )
    parser.add_argument(
        "--preprocessed-cc",
        default=project_root / "data" / "RepoRT_RP" / "preprocessed_data" / "preprocessed_cc_data.tsv",
        type=Path,
    )
    parser.add_argument(
        "--preprocessed-grad",
        default=project_root / "data" / "RepoRT_RP" / "preprocessed_data" / "preprocessed_gradient_data.tsv",
        type=Path,
    )
    parser.add_argument(
        "--column-properties",
        default=project_root / "data" / "benchmarks" / "graphormer_rt" / "column_properties_from_upstream_sample.tsv",
        type=Path,
    )
    parser.add_argument("--molecule-id-col", default="molecule_id")
    parser.add_argument("--smiles-col", default="smiles.std")
    parser.add_argument("--rt-col", default="rt")
    parser.add_argument("--gradient-time-scale", default=1.0 / 60.0, type=float)
    parser.add_argument("--missing-column-properties", choices=["zero", "error"], default="zero")
    parser.add_argument("--missing-metadata-policy", choices=["exclude", "error"], default="error")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_assets(args)
    print(f"output_dir: {args.output_dir}")
    print(f"rows: {summary['n_rows']}")
    print(f"methods: {summary['n_methods']}")
    print(f"folds: {summary['n_folds']}")


if __name__ == "__main__":
    main()
