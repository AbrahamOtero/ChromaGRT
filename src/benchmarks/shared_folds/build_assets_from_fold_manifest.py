"""Build ChromaGRT and GraphormerRT assets from a versioned fold manifest.

The exact base-fold assignments used in the article are provided in the repository under 
fold_manifests/article-v1. Separate tab-separated files are supplied for the random, 
Bemis--Murcko scaffold, and chromatographic-condition scenarios: random_10fold.tsv, bm_scaffold_10fold.tsv, 
and cc_10fold.tsv. Each file contains one row for every observation in the curated dataset 
and four columns: cc_id, molecule_id, inchi.std, and base\_fold. 

"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from src.benchmarks.shared_folds.asset_generation import (
    _add_graphormer_rt_columns,
    _build_metadata,
    _write_chromagrt_splits,
    _write_graphormer_rt_csv,
    _write_graphormer_rt_manifests,
    _write_validation_report,
)
from src.benchmarks.shared_folds.export_fold_manifests import KEY_COLUMNS


SCENARIO_PREFIXES = {
    "random": "rkf",
    "bm_scaffold": "bmkf",
    "cc": "full",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_tsv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError(f"{path} contains no observations")
    return frame


def apply_fold_manifest(dataset: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Attach exact base-fold assignments after a one-to-one key match."""
    required_manifest_columns = (*KEY_COLUMNS, "base_fold")
    missing_dataset = set(KEY_COLUMNS) - set(dataset.columns)
    missing_manifest = set(required_manifest_columns) - set(manifest.columns)
    if missing_dataset:
        raise ValueError(f"Dataset is missing manifest key columns: {', '.join(sorted(missing_dataset))}")
    if missing_manifest:
        raise ValueError(f"Fold manifest is missing required columns: {', '.join(sorted(missing_manifest))}")
    if (dataset.loc[:, list(KEY_COLUMNS)] == "").any(axis=None):
        raise ValueError("Dataset contains an empty manifest key")
    if (manifest.loc[:, list(KEY_COLUMNS)] == "").any(axis=None):
        raise ValueError("Fold manifest contains an empty observation key")
    if dataset.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Dataset contains duplicate manifest keys")
    if manifest.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Fold manifest contains duplicate observation keys")

    merged = dataset.merge(
        manifest.loc[:, [*KEY_COLUMNS, "base_fold"]],
        on=list(KEY_COLUMNS),
        how="outer",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    missing_manifest_rows = int((merged["_merge"] == "left_only").sum())
    missing_dataset_rows = int((merged["_merge"] == "right_only").sum())
    if missing_manifest_rows or missing_dataset_rows:
        raise ValueError(
            "Dataset and fold manifest do not identify the same observations: "
            f"{missing_manifest_rows} dataset-only, {missing_dataset_rows} manifest-only."
        )
    merged = merged.drop(columns="_merge")
    return merged


def _validate_folds(rows: pd.DataFrame, scenario: str, n_folds: int) -> None:
    numeric_folds = pd.to_numeric(rows["base_fold"], errors="raise")
    if not (numeric_folds % 1 == 0).all():
        raise ValueError("Fold manifest contains a non-integer base_fold")
    rows["fold"] = numeric_folds.astype(int)
    expected = set(range(n_folds))
    actual = set(rows["fold"])
    allowed = expected | ({-1} if scenario == "bm_scaffold" else set())
    invalid = actual - allowed
    if invalid:
        raise ValueError(f"Fold manifest has unsupported assignments: {sorted(invalid)}")
    if expected - actual:
        raise ValueError(f"Fold manifest is missing assignments: {sorted(expected - actual)}")
    if scenario == "bm_scaffold" and -1 not in actual:
        raise ValueError("BM scaffold manifest is missing its always-training assignment (-1)")


def _order_rows_for_asset_generation(rows: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario != "cc":
        return rows
    return rows.sort_values("fold", kind="stable")


def build_assets(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")

    dataset = _read_tsv(
        args.input_path,
        (args.cc_col, args.molecule_id_col, args.smiles_col, args.rt_col, *KEY_COLUMNS),
    )
    manifest = _read_tsv(args.fold_manifest, (*KEY_COLUMNS, "base_fold"))
    rows = apply_fold_manifest(dataset, manifest)
    _validate_folds(rows, args.scenario, args.n_folds)
    rows = rows.drop(columns="base_fold")
    rows = _order_rows_for_asset_generation(rows, args.scenario)

    rows = _add_graphormer_rt_columns(
        rows,
        args.molecule_id_col,
        args.smiles_col,
        args.rt_col,
        SCENARIO_PREFIXES[args.scenario],
    )
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
        raise ValueError(
            "The supplied data can not reproduce the assets because metadata "
            f"are missing for GraphormerRT methods: {sorted(missing_methods)[:20]}"
        )

    rows = rows.reset_index(drop=True)
    rows["row_index"] = rows.index
    rows["sample_id"] = rows["row_index"].map(
        lambda value: f"{SCENARIO_PREFIXES[args.scenario]}_{int(value):06d}"
    )
    args.output_dir.mkdir(parents=True)
    _write_graphormer_rt_csv(rows, args.output_dir)
    with (args.output_dir / "RP_metadata.pickle").open("wb") as handle:
        pickle.dump(metadata, handle)
    rows.to_csv(args.output_dir / "master_manifest.csv", index=False)
    metadata_report.to_csv(args.output_dir / "metadata_coverage.tsv", sep="\t", index=False)
    graphormer_rt_counts = _write_graphormer_rt_manifests(rows, args.output_dir, args.n_folds)
    chromagrt_counts = _write_chromagrt_splits(rows, args.output_dir, args.n_folds)
    _write_validation_report(
        rows,
        args.output_dir,
        args.n_folds,
        args.scaffold_col if args.scenario == "bm_scaffold" and args.scaffold_col in rows.columns else None,
        args.cc_col,
    )

    summary = {
        "scenario": args.scenario,
        "fold_manifest": args.fold_manifest.name,
        "n_folds": args.n_folds,
        "n_rows": int(len(rows)),
        "n_cc_ids": int(rows[args.cc_col].nunique()),
        "n_methods": int(rows["graphormer_rt_method_id"].nunique()),
        "graphormer_rt_split_counts": graphormer_rt_counts,
        "chromagrt_split_counts": chromagrt_counts,
        "split_policy": "for run i: test=fold_i, valid=fold_(i+1)%K, train=all remaining folds; BM fold=-1 rows are always train",
    }
    with (args.output_dir / "manifest_application_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIO_PREFIXES))
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--fold-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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
    parser.add_argument("--cc-col", default="cc_id")
    parser.add_argument("--scaffold-col", default="ms_smiles")
    parser.add_argument("--n-folds", default=10, type=int)
    parser.add_argument("--gradient-time-scale", default=1.0 / 60.0, type=float)
    parser.add_argument("--missing-column-properties", choices=["zero", "error"], default="zero")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_assets(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
