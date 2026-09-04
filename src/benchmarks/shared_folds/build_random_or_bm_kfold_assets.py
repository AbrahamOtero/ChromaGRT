"""Build ChromaGRT 10-fold assets for random or Bemis-Murcko evaluation.

The output always contains ChromaGRT split tables.  

- chromagrt_splits/fold_*/train_data.tsv.

"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarks.shared_folds.asset_generation import (
    NO_MURCKO_SCAFFOLD,
    SPLIT_FILES,
    _add_graphormer_rt_columns,
    _build_metadata,
    _method_from_molecule_id,
    _write_chromagrt_splits,
    _write_graphormer_rt_csv,
    _write_graphormer_rt_manifests,
    _write_validation_report,
)


def _normalize_scaffold_value(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return NO_MURCKO_SCAFFOLD
    return str(value).strip()


def add_or_normalize_murcko_scaffold_column(
    df: pd.DataFrame,
    smiles_col: str,
    scaffold_col: str,
    suppress_rdkit_warnings: bool,
) -> pd.DataFrame:
    """Return a copy with a normalized Bemis--Murcko scaffold column."""
    result = df.copy()
    if scaffold_col in result.columns:
        result[scaffold_col] = result[scaffold_col].map(_normalize_scaffold_value)
        return result
    if smiles_col not in result.columns:
        raise ValueError(f"Cannot build {scaffold_col}: missing input column {smiles_col}.")

    from rdkit import RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    if suppress_rdkit_warnings:
        RDLogger.DisableLog("rdApp.warning")

    def murcko_from_smiles(smiles: object) -> str:
        if pd.isna(smiles) or not str(smiles).strip():
            return NO_MURCKO_SCAFFOLD
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmilesFromSmiles(str(smiles))
        except Exception:
            scaffold = ""
        return _normalize_scaffold_value(scaffold)

    position = result.columns.get_loc(smiles_col)
    result.insert(position + 1, scaffold_col, result[smiles_col].map(murcko_from_smiles))
    return result


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _assign_random_folds(df: pd.DataFrame, n_folds: int, seed: int, cc_col: str) -> tuple[pd.Series, dict]:
    fold_ids = pd.Series(index=df.index, dtype="int64")
    rng = np.random.default_rng(seed)
    for _, group in df.groupby(cc_col, sort=False):
        positions = group.index.to_numpy().copy()
        rng.shuffle(positions)
        for offset, row_index in enumerate(positions):
            fold_ids.loc[row_index] = offset % n_folds
    fold_ids = fold_ids.astype(int)
    return fold_ids, {
        "strategy": "random_stratified_by_cc_id",
        "seed": int(seed),
        "n_folds": int(n_folds),
        "stratification_column": cc_col,
    }


def _assign_bm_scaffold_folds(
    df: pd.DataFrame,
    n_folds: int,
    seed: int,
    scaffold_col: str,
) -> tuple[pd.Series, dict]:
    non_empty_df = df[df[scaffold_col] != NO_MURCKO_SCAFFOLD].copy()
    no_scaffold_df = df[df[scaffold_col] == NO_MURCKO_SCAFFOLD].copy()
    scaffold_counts = non_empty_df[scaffold_col].value_counts(sort=False)

    rng = np.random.default_rng(seed)
    tie_breakers = dict(zip(scaffold_counts.index, rng.random(len(scaffold_counts))))
    ordered_scaffolds = sorted(
        scaffold_counts.index,
        key=lambda scaffold: (-int(scaffold_counts[scaffold]), tie_breakers[scaffold]),
    )

    fold_sizes = [0] * n_folds
    scaffold_to_fold: dict[str, int] = {}
    for scaffold in ordered_scaffolds:
        size = int(scaffold_counts[scaffold])
        fold_idx = min(range(n_folds), key=lambda idx: (fold_sizes[idx], idx))
        scaffold_to_fold[scaffold] = fold_idx
        fold_sizes[fold_idx] += size

    fold_ids = pd.Series(index=df.index, dtype="int64")
    fold_ids.loc[non_empty_df.index] = non_empty_df[scaffold_col].map(scaffold_to_fold)
    fold_ids.loc[no_scaffold_df.index] = -1
    fold_ids = fold_ids.astype(int)
    return fold_ids, {
        "strategy": "bemis_murcko_scaffold_grouped",
        "seed": int(seed),
        "n_folds": int(n_folds),
        "scaffold_column": scaffold_col,
        "empty_murcko_rows_assigned_to_train_each_run": int(len(no_scaffold_df)),
        "pseudo_scaffold_for_empty_murcko": NO_MURCKO_SCAFFOLD,
        "rows_with_non_empty_murcko": int(len(non_empty_df)),
        "unique_non_empty_scaffolds": int(non_empty_df[scaffold_col].nunique()),
        "base_fold_sizes_non_empty_only": {f"fold_{idx}": int(size) for idx, size in enumerate(fold_sizes)},
    }


def build_assets(args: argparse.Namespace) -> dict:
    if args.refuse_existing_output and args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    df = pd.read_csv(args.input_path, sep="\t", dtype=str).fillna("")
    sample_prefix = "rkf" if args.strategy == "random" else "bmkf"

    if args.strategy == "random":
        fold_ids, split_metadata = _assign_random_folds(df, args.n_folds, args.seed, args.cc_col)
        scaffold_col_for_report = None
    elif args.strategy == "bm_scaffold":
        df = add_or_normalize_murcko_scaffold_column(
            df,
            smiles_col=args.smiles_col,
            scaffold_col=args.scaffold_col,
            suppress_rdkit_warnings=not args.show_rdkit_warnings,
        )
        fold_ids, split_metadata = _assign_bm_scaffold_folds(df, args.n_folds, args.seed, args.scaffold_col)
        scaffold_col_for_report = args.scaffold_col
    else:
        raise ValueError(f"Unsupported strategy: {args.strategy}")

    df = df.copy()
    df["fold"] = fold_ids
    rows = _add_graphormer_rt_columns(df, args.molecule_id_col, args.smiles_col, args.rt_col, sample_prefix)
    write_graphormer_rt_assets = getattr(args, "write_graphormer_rt_assets", False)

    if write_graphormer_rt_assets:
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
    else:
        metadata = None
        metadata_report = None
        missing_methods = set()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = rows.reset_index(drop=True)
    rows["row_index"] = rows.index
    rows["sample_id"] = rows["row_index"].map(lambda value: f"{sample_prefix}_{int(value):06d}")

    rows.to_csv(args.output_dir / "master_manifest.csv", index=False)
    chromagrt_counts = _write_chromagrt_splits(rows, args.output_dir, args.n_folds)
    _write_validation_report(rows, args.output_dir, args.n_folds, scaffold_col_for_report, args.cc_col)

    if write_graphormer_rt_assets:
        _write_graphormer_rt_csv(rows, args.output_dir)
        with (args.output_dir / "RP_metadata.pickle").open("wb") as handle:
            pickle.dump(metadata, handle)
        metadata_report.to_csv(args.output_dir / "metadata_coverage.tsv", sep="\t", index=False)
        graphormer_rt_counts = _write_graphormer_rt_manifests(rows, args.output_dir, args.n_folds)
        pd.DataFrame(
            [{"fold": key, **value} for key, value in sorted(graphormer_rt_counts.items())]
        ).to_csv(args.output_dir / "fold_split_summary.tsv", sep="\t", index=False)

    summary = {
        "strategy": args.strategy,
        "input_path": str(args.input_path.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "n_folds": int(args.n_folds),
        "n_rows": int(len(rows)),
        "n_methods": int(rows["graphormer_rt_method_id"].nunique()),
        "n_cc_ids": int(rows[args.cc_col].nunique()) if args.cc_col in rows.columns else None,
        "graphormer_rt_assets": write_graphormer_rt_assets,
        "base_fold_counts": {
            str(int(fold)): int(count)
            for fold, count in rows["fold"].value_counts(sort=False).sort_index().items()
        },
        "split_metadata": split_metadata,
        "chromagrt_split_counts": chromagrt_counts,
        "split_policy": "for run i: test=fold_i, valid=fold_(i+1)%K, train=all remaining folds; fold=-1 rows are always train",
    }
    if write_graphormer_rt_assets:
        summary.update(
            {
                "n_metadata_methods": int(len(metadata) - 1),
                "n_missing_metadata_methods": int(len(missing_methods)),
                "missing_metadata_policy": args.missing_metadata_policy,
                "gradient_time_scale": args.gradient_time_scale,
                "metadata_encoding": "GraphormerRT-compatible gradient block",
                "graphormer_rt_split_counts": graphormer_rt_counts,
            }
        )
    with (args.output_dir / "mapping_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    project_root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["random", "bm_scaffold"], required=True)
    parser.add_argument(
        "--input-path",
        default=project_root
        / "data"
        / "RepoRT_RP"
        / "processed_data"
        / "no_SMRT"
        / "complete_processed_data_with_tanaka.tsv",
        type=Path,
    )
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
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gradient-time-scale", default=1.0 / 60.0, type=float)
    parser.add_argument("--missing-column-properties", choices=["zero", "error"], default="zero")
    parser.add_argument("--missing-metadata-policy", choices=["exclude", "error"], default="error")
    parser.add_argument(
        "--write-graphormer-rt-assets",
        action="store_true",
        help="Also write GraphormerRT inputs and per-fold manifests.",
    )
    parser.add_argument(
        "--refuse-existing-output",
        action="store_true",
        help="Fail if --output-dir already exists instead of reusing and overwriting its contents.",
    )
    parser.add_argument("--show-rdkit-warnings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_assets(args)
    print(f"output_dir: {args.output_dir}")
    print(f"strategy: {summary['strategy']}")
    print(f"rows: {summary['n_rows']}")
    print(f"methods: {summary['n_methods']}")
    print(f"folds: {summary['n_folds']}")


if __name__ == "__main__":
    main()
