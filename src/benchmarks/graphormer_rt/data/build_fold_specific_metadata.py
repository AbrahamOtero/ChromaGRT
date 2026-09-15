"""Build GraphormerRT metadata dictionaries with training-fold imputation.

The shared benchmark assets contain one GraphormerRT metadata dictionary for
compatibility with the original workflow.  This module leaves those assets
unchanged and writes one self-contained GraphormerRT data view per fold.  Each
view reuses the common ``report_rp.csv`` but has its own ``RP_metadata.pickle``
whose physical column fields are imputed from that fold's training conditions.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from src.benchmarks.graphormer_rt.data.build_report_rp_inputs import (
    METADATA_HEADER,
    _as_str_number,
)


PHYSICAL_COLUMNS = (
    "column.length",
    "column.id",
    "column.particle.size",
    "column.temperature",
    "column.flowrate",
)
T0_COLUMN = "column.t0"
METHOD_COLUMN = "graphormer_rt_method_id"
CONDITION_COLUMN = "cc_id"
SPLIT_COLUMN = "split"

PICKLE_HEADER_BY_COLUMN = {
    "column.length": "col_length",
    "column.id": "col_innerdiam",
    "column.particle.size": "col_part_size",
    "column.temperature": "temp",
    "column.flowrate": "col_fl",
    "column.t0": "col_dead",
}


def _read_split_table(path: Path) -> pd.DataFrame:
    required = {METHOD_COLUMN, CONDITION_COLUMN, SPLIT_COLUMN, *PHYSICAL_COLUMNS, T0_COLUMN}
    frame = pd.read_csv(path, sep="\t", dtype=str)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    actual_splits = set(frame[SPLIT_COLUMN].dropna())
    expected_splits = {"train", "valid", "test"}
    if actual_splits != expected_splits:
        raise ValueError(f"{path} has splits {sorted(actual_splits)}, expected {sorted(expected_splits)}")
    if frame[METHOD_COLUMN].isna().any() or frame[CONDITION_COLUMN].isna().any():
        raise ValueError(f"{path} has missing method or condition identifiers")
    return frame


def _fit_training_means(train_df: pd.DataFrame) -> tuple[dict[str, float], int]:
    """Fit the ChromaGRT-equivalent numerical means from training conditions."""
    condition_df = train_df.drop_duplicates(subset=[CONDITION_COLUMN])
    means: dict[str, float] = {}
    for column in PHYSICAL_COLUMNS:
        values = pd.to_numeric(condition_df[column], errors="coerce")
        mean = values.mean(skipna=True)
        if pd.isna(mean):
            raise ValueError(f"Cannot impute {column}: no observed training value is available")
        means[column] = round(float(mean), 2)
    return means, int(len(condition_df))


def _apply_physical_imputation(frame: pd.DataFrame, means: dict[str, float]) -> pd.DataFrame:
    """Apply the numerical metadata and t0 rules used for one GraphormerRT fold."""
    result = frame.copy()
    for column, mean in means.items():
        values = pd.to_numeric(result[column], errors="coerce")
        result[column] = values.fillna(mean)

    t0 = pd.to_numeric(result[T0_COLUMN], errors="coerce")
    result[T0_COLUMN] = t0
    needs_proxy = t0.isna() | (t0 == 0)
    if needs_proxy.any():
        diameter = pd.to_numeric(result.loc[needs_proxy, "column.id"], errors="coerce")
        length = pd.to_numeric(result.loc[needs_proxy, "column.length"], errors="coerce")
        flow = pd.to_numeric(result.loc[needs_proxy, "column.flowrate"], errors="coerce")
        if diameter.isna().any() or length.isna().any() or flow.isna().any() or (flow <= 0).any():
            raise ValueError("Cannot calculate column.t0 after training-fold metadata imputation")
        proxy = ((0.66 * np.pi * (diameter / 2) ** 2 * length / 10) / flow / 100).round(5)
        result.loc[needs_proxy, T0_COLUMN] = proxy
    unresolved = [
        column
        for column in (*PHYSICAL_COLUMNS, T0_COLUMN)
        if pd.to_numeric(result[column], errors="coerce").isna().any()
    ]
    if unresolved:
        raise ValueError(f"Unresolved metadata after training-fold imputation: {unresolved}")
    return result


def _method_values(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return one validated physical-column vector for each method."""
    values_by_method: dict[str, dict[str, float]] = {}
    for method_id, method_df in frame.groupby(METHOD_COLUMN, sort=True):
        if method_df[CONDITION_COLUMN].nunique(dropna=False) != 1:
            raise ValueError(f"Method {method_id} maps to multiple chromatographic conditions")
        values: dict[str, float] = {}
        for column in (*PHYSICAL_COLUMNS, T0_COLUMN):
            numeric = pd.to_numeric(method_df[column], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(numeric, numeric[0], rtol=0.0, atol=0.0):
                raise ValueError(f"Method {method_id} has inconsistent values for {column}")
            values[column] = float(numeric[0])
        values_by_method[str(method_id)] = values
    return values_by_method


def _metadata_positions(metadata: dict[str, list[Any]]) -> dict[str, int]:
    if "1" not in metadata:
        raise ValueError("Base GraphormerRT metadata dictionary has no header entry '1'")
    header = metadata["1"]
    if header != METADATA_HEADER:
        raise ValueError("Base GraphormerRT metadata header differs from the expected GraphormerRT schema")
    return {
        column: header.index(header_name)
        for column, header_name in PICKLE_HEADER_BY_COLUMN.items()
    }


def _update_metadata(
    base_metadata: dict[str, list[Any]],
    values_by_method: dict[str, dict[str, float]],
) -> dict[str, list[Any]]:
    positions = _metadata_positions(base_metadata)
    base_methods = set(base_metadata) - {"1"}
    data_methods = set(values_by_method)
    if base_methods != data_methods:
        missing = sorted(data_methods - base_methods)
        extra = sorted(base_methods - data_methods)
        raise ValueError(
            "Methods in GraphormerRT metadata and split data differ: "
            f"{len(missing)} absent from metadata, {len(extra)} absent from data"
        )

    updated = copy.deepcopy(base_metadata)
    for method_id, values in values_by_method.items():
        vector = updated[method_id]
        if len(vector) != len(METADATA_HEADER):
            raise ValueError(f"Method {method_id} has a malformed GraphormerRT metadata vector")
        for column, position in positions.items():
            value = values[column]
            vector[position] = float(value) if column == "column.flowrate" else _as_str_number(value)
    return updated


def _count_imputed_values(frame: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for column in PHYSICAL_COLUMNS:
        counts[column] = int(pd.to_numeric(frame[column], errors="coerce").isna().sum())
    t0 = pd.to_numeric(frame[T0_COLUMN], errors="coerce")
    counts[T0_COLUMN] = int((t0.isna() | (t0 == 0)).sum())
    return counts


def build_fold_specific_metadata(asset_dir: Path, output_dir: Path, n_folds: int) -> dict[str, Any]:
    """Build one GraphormerRT data view per predefined fold without changing assets."""
    asset_dir = Path(asset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    report_path = asset_dir / "report_rp.csv"
    metadata_path = asset_dir / "RP_metadata.pickle"
    if not report_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Asset directory must contain report_rp.csv and RP_metadata.pickle")
    with metadata_path.open("rb") as handle:
        base_metadata = pickle.load(handle)
    if not isinstance(base_metadata, dict):
        raise TypeError("Base GraphormerRT metadata must be a dictionary")
    _metadata_positions(base_metadata)

    fold_products = []
    for fold_index in range(n_folds):
        split_path = asset_dir / "chromagrt_splits" / f"fold_{fold_index}" / "complete_manifest_dataset.tsv"
        frame = _read_split_table(split_path)
        means, n_training_conditions = _fit_training_means(frame[frame[SPLIT_COLUMN] == "train"])
        values_by_method = _method_values(_apply_physical_imputation(frame, means))
        metadata = _update_metadata(base_metadata, values_by_method)
        fold_products.append(
            {
                "fold": f"fold_{fold_index}",
                "metadata": metadata,
                "means": means,
                "training_conditions": n_training_conditions,
                "imputed_values": _count_imputed_values(frame),
            }
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    for product in fold_products:
        fold_dir = temporary_dir / product["fold"]
        fold_dir.mkdir()
        os.link(report_path, fold_dir / "report_rp.csv")
        with (fold_dir / "RP_metadata.pickle").open("wb") as handle:
            pickle.dump(product["metadata"], handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "asset_dir": str(asset_dir),
        "base_metadata": str(metadata_path.resolve()),
        "common_report_rp": str(report_path.resolve()),
        "metadata_policy": (
            "Physical column metadata are imputed from one row per training cc_id; "
            "the same fold-specific values are used for train, validation, and test."
        ),
        "n_folds": n_folds,
        "folds": [
            {
                "fold": product["fold"],
                "training_conditions": product["training_conditions"],
                "means": product["means"],
                "imputed_values": product["imputed_values"],
            }
            for product in fold_products
        ],
    }
    with (temporary_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_dir, output_dir)

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-folds", default=10, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or args.asset_dir / "graphormer_rt_fold_data"
    summary = build_fold_specific_metadata(args.asset_dir, output_dir, args.n_folds)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
