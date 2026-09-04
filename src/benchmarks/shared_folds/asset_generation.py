"""Shared code for writing ChromaGRT and GraphormerRT fold assets.

The public builders use this module for the common representation of rows,
chromatographic metadata, and train/validation/test files.  Fold-assignment
strategies are in their respective builder modules.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pandas as pd

from src.benchmarks.graphormer_rt.data.build_report_rp_inputs import (
    METADATA_HEADER,
    _load_column_properties,
    _metadata_for_method,
    _normalise_method_id,
    _to_float,
    _value,
)


SPLIT_FILES = {
    "train": "train_data.tsv",
    "valid": "val_data.tsv",
    "test": "test_data.tsv",
}
NO_MURCKO_SCAFFOLD = "__NO_MURCKO_SCAFFOLD__"


def _method_from_molecule_id(value: str) -> str:
    return _normalise_method_id(str(value).split("_", 1)[0])


def _split_for_fold(row_fold: int, test_fold: int, n_folds: int) -> str:
    if row_fold < 0:
        return "train"
    if row_fold == test_fold:
        return "test"
    if row_fold == (test_fold + 1) % n_folds:
        return "valid"
    return "train"


def _scale_gradient_times(row: pd.Series, scale: float) -> pd.Series:
    row = row.copy().astype(object)
    for idx in range(18):
        column = f"t [min]_{idx}"
        if column in row and str(row[column]) != "":
            row[column] = _to_float(row[column]) * scale
    return row


def _as_graphormer_rt_number(value: float) -> str:
    if float(value).is_integer():
        return f"{float(value):.1f}"
    return f"{float(value):g}"


def _gradient_points(row: pd.Series, organic_phase: str = "B") -> list[tuple[float, float]]:
    points = []
    for idx in range(18):
        t = _value(row, f"t [min]_{idx}", "")
        b = _value(row, f"{organic_phase} [%]_{idx}", "")
        if t == "" or b == "":
            continue
        points.append((_to_float(t), _to_float(b)))
    return points


def _organic_phase_from_gradient(row: pd.Series) -> str:
    a0 = _to_float(_value(row, "A [%]_0")) 
    b0 = _to_float(_value(row, "B [%]_0"))
    return "A" if b0 > 50 and a0 < b0 else "B"


def _graphormer_rt_gradient_block(grad_row: pd.Series) -> list[str]:
    """Replicate the GraphormerRT sample-style gradient block."""
    organic_phase = _organic_phase_from_gradient(grad_row)
    points = _gradient_points(grad_row, organic_phase=organic_phase)
    if not points:
        return ["0.0"] * 8

    max_index = max(range(len(points)), key=lambda idx: points[idx][1])
    inflections: list[tuple[float, float]] =  [] 
    for idx in range(1, len(points)):
        if points[idx][1] != points[idx - 1][1] and idx <= max_index:
            inflections.append(points[idx - 1])
            inflections.append(points[idx])
            inflections = list(set(inflections))

    values = [_as_graphormer_rt_number(points[0][0]), _as_graphormer_rt_number(points[0][1])]
    previous_time = -1.0
    count = 0
    while count < len(inflections):
        time, _ = inflections[count]
        if abs(time - previous_time) < 0.3:
            inflections.pop(count)
            continue
        previous_time = time
        count += 1

    for time, percent_b in inflections:
        values.extend([_as_graphormer_rt_number(time), _as_graphormer_rt_number(percent_b)])
    values.extend(["0.0"] * (8 - len(values)))
    return values[:8]


def _build_metadata(
    method_ids: set[str],
    preprocessed_cc: Path,
    preprocessed_grad: Path,
    column_properties: Path | None,
    gradient_time_scale: float,
    missing_column_properties: str,
) -> tuple[dict[str, list], pd.DataFrame]:
    cc_df = pd.read_csv(preprocessed_cc, sep="\t", dtype=str).fillna("")
    grad_df = pd.read_csv(preprocessed_grad, sep="\t", dtype=str).fillna("")
    cc_df["dir_id"] = cc_df["dir_id"].map(_normalise_method_id)
    grad_df["dir_id"] = grad_df["dir_id"].map(_normalise_method_id)
    cc_by_id = {row["dir_id"]: row for _, row in cc_df.iterrows()}
    grad_by_id = {row["dir_id"]: row for _, row in grad_df.iterrows()}
    properties_by_dir_id, properties_by_column_key = _load_column_properties(column_properties)

    metadata = {"1": METADATA_HEADER}
    report_rows = []
    for method_id in sorted(method_ids):
        status = "ok"
        property_source = ""
        if method_id not in cc_by_id or method_id not in grad_by_id:
            status = "missing_cc_or_gradient"
        else:
            grad_row = _scale_gradient_times(grad_by_id[method_id], gradient_time_scale)
            values, property_source = _metadata_for_method(
                method_id,
                cc_by_id[method_id],
                grad_row,
                properties_by_dir_id,
                properties_by_column_key,
                missing_column_properties,
            )
            values[11:19] = _graphormer_rt_gradient_block(grad_row)
            metadata[method_id] = values
        report_rows.append({"method_id": method_id, "status": status, "property_source": property_source})

    return metadata, pd.DataFrame(report_rows)


def _add_graphormer_rt_columns(
    df: pd.DataFrame,
    molecule_id_col: str,
    smiles_col: str,
    rt_col: str,
    sample_prefix: str,
) -> pd.DataFrame:
    records = []
    for row_index, (_, row) in enumerate(df.iterrows()):
        method_id = _method_from_molecule_id(row[molecule_id_col])
        records.append(
            {
                **row.to_dict(),
                "sample_id": f"{sample_prefix}_{row_index:06d}",
                "row_index": row_index,
                "graphormer_rt_method_id": method_id,
                "source_row_index": int(row.name),
                "graphormer_rt_smiles": str(row[smiles_col]).replace("Q", "#"),
                "rt_seconds": float(row[rt_col]),
                "rt_minutes": float(row[rt_col]) / 60.0,
            }
        )
    return pd.DataFrame(records)


def _write_graphormer_rt_csv(rows: pd.DataFrame, output_dir: Path) -> None:
    with (output_dir / "report_rp.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for _, row in rows.iterrows():
            writer.writerow([row["graphormer_rt_method_id"], row["graphormer_rt_smiles"], f"{row['rt_minutes']:.8g}"])


def _write_graphormer_rt_manifests(rows: pd.DataFrame, output_dir: Path, n_folds: int) -> dict[str, dict[str, int]]:
    manifest_dir = output_dir / "graphormer_rt_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    split_counts_by_fold = {}
    for test_fold in range(n_folds):
        split_counts = Counter()
        manifest_rows = []
        for _, row in rows.iterrows():
            split = _split_for_fold(int(row["fold"]), test_fold, n_folds)
            split_counts[split] += 1
            manifest_rows.append({
                    "sample_id": row["sample_id"],
                    "row_index": row["row_index"],
                    "method_id": row["graphormer_rt_method_id"],
                    "smiles": row["graphormer_rt_smiles"],
                    "rt": f"{row['rt_minutes']:.8g}",
                    "split": split,
                }
            )
        with (manifest_dir / f"fold_{test_fold}_graphormer_rt_manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_id", "row_index", "method_id", "smiles", "rt", "split"])
            writer.writeheader()
            writer.writerows(manifest_rows)
        split_counts_by_fold[f"fold_{test_fold}"] = dict(sorted(split_counts.items()))
    return split_counts_by_fold


def _write_chromagrt_splits(
    rows: pd.DataFrame,
    output_dir: Path,
    n_folds: int,
    include_source_fold_file: bool = False,
) -> dict[str, dict[str, int]]:
    splits_root = output_dir / "chromagrt_splits"
    splits_root.mkdir(parents=True, exist_ok=True)
    metadata_cols = [
        "sample_id",
        "row_index",
        "split",
        "graphormer_rt_method_id",
        "source_fold",
        *( ["source_fold_file"] if include_source_fold_file else [] ),
        "source_row_index",
    ]
    excluded = {
        "sample_id",
        "row_index",
        "split",
        "fold",
        "source_fold",
        "source_fold_file",
        "source_row_index",
        "graphormer_rt_method_id",
        "graphormer_rt_smiles",
        "rt_seconds",
        "rt_minutes",
    }
    source_cols = [column for column in rows.columns if column not in excluded]
    output_cols = metadata_cols + source_cols
    split_counts_by_fold = {}

    for test_fold in range(n_folds):
        fold_dir = splits_root / f"fold_{test_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_df = rows.copy()
        fold_df["split"] = fold_df["fold"].astype(int).map(lambda value: _split_for_fold(value, test_fold, n_folds))
        fold_df["source_fold"] = fold_df["fold"]
        split_counts = {}
        for split, filename in SPLIT_FILES.items():
            split_df = fold_df[fold_df["split"] == split].copy()
            split_counts[split] = int(len(split_df))
            split_df[output_cols].to_csv(fold_dir / filename, sep="\t", index=False)
        fold_df[output_cols].to_csv(fold_dir / "complete_manifest_dataset.tsv", sep="\t", index=False)
        split_counts_by_fold[f"fold_{test_fold}"] = split_counts
    return split_counts_by_fold


def _write_validation_report(
    rows: pd.DataFrame,
    output_dir: Path,
    n_folds: int,
    scaffold_col: str | None,
    cc_col: str,
) -> None:
    report_rows = []
    for test_fold in range(n_folds):
        fold_df = rows.copy()
        fold_df["split"] = fold_df["fold"].astype(int).map(lambda value: _split_for_fold(value, test_fold, n_folds))
        for split in SPLIT_FILES:
            split_df = fold_df[fold_df["split"] == split]
            report_rows.append(
                {
                    "fold": f"fold_{test_fold}",
                    "split": split,
                    "molecule_count": int(len(split_df)),
                    "cc_count": int(split_df[cc_col].nunique()) if cc_col in split_df.columns else None,
                    "scaffold_count": int(split_df[scaffold_col].nunique()) if scaffold_col and scaffold_col in split_df.columns else None,
                    "no_murcko_scaffold_count": int((split_df[scaffold_col] == NO_MURCKO_SCAFFOLD).sum())
                    if scaffold_col and scaffold_col in split_df.columns
                    else None,
                }
            )
        if scaffold_col and scaffold_col in rows.columns:
            split_sets = {
                split: set(fold_df.loc[fold_df["split"] == split, scaffold_col])
                for split in SPLIT_FILES
            }
            for left, right in [("train", "valid"), ("train", "test"), ("valid", "test")]:
                shared = split_sets[left].intersection(split_sets[right])
                report_rows.append(
                    {
                        "fold": f"fold_{test_fold}",
                        "split": f"{left}_vs_{right}",
                        "molecule_count": None,
                        "cc_count": None,
                        "scaffold_count": len(shared),
                        "no_murcko_scaffold_count": int(NO_MURCKO_SCAFFOLD in shared),
                    }
                )
    pd.DataFrame(report_rows).to_csv(output_dir / "kfold_validation_report.tsv", sep="\t", index=False)
