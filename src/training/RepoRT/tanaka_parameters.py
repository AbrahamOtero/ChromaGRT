"""Add Tanaka column descriptors to curated RepoRT data."""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd


TANAKA_VALUE_COLUMNS = [
    "tanaka_kpb",
    "tanaka_alpha_ch2",
    "tanaka_alpha_t_o",
    "tanaka_alpha_c_p",
    "tanaka_alpha_b_p",
    "tanaka_alpha_b_p_1",
]
TANAKA_MASK_COLUMNS = [f"{column}_is_missing" for column in TANAKA_VALUE_COLUMNS]
ARTICLE_CURATED_TANAKA_PATH = Path(__file__).resolve().parent / "resources" / "article_curated_tanaka.tsv"


def has_nonzero_tanaka(values):
    for value in values:
        if pd.isna(value):
            continue
        try:
            if float(value) != 0.0:
                return True
        except (TypeError, ValueError):
            if str(value).strip() not in {"", "0", "0.0"}:
                return True
    return False


def normalize_column_key(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_dir_ids(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item).zfill(4) for item in value]
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        parsed = [value]
    if not isinstance(parsed, list):
        parsed = [parsed]
    return [str(item).zfill(4) for item in parsed]


def join_unique(values):
    clean_values = sorted({str(value).strip() for value in values if not pd.isna(value) and str(value).strip()})
    return "|".join(clean_values)


def load_raw_column_metadata(raw_cc_path, cc_map_path):
    raw_cc = pd.read_csv(raw_cc_path, sep="\t", dtype=str)
    cc_map = pd.read_csv(cc_map_path, sep="\t", dtype=str)
    cc_map = cc_map.assign(dir_id=cc_map["dir_ids"].map(parse_dir_ids)).explode("dir_id")

    metadata_columns = ["dir_id", "column.name", "column.id", "column.usp.code"]
    metadata = cc_map.merge(raw_cc.loc[:, metadata_columns], on="dir_id", how="left")
    metadata = (
        metadata.groupby("cc_id", as_index=False)
        .agg(
            {
                "column.name": join_unique,
                "column.id": join_unique,
                "column.usp.code": join_unique,
            }
        )
    )
    metadata["column_key"] = metadata["column.name"].map(normalize_column_key)
    metadata.loc[metadata["column_key"] == "", "column_key"] = metadata.loc[
        metadata["column_key"] == "", "cc_id"
    ]
    return metadata


def _insert_columns_before(df, anchor_column, columns):
    if anchor_column not in df.columns:
        raise ValueError(f"Can not insert before missing column: {anchor_column}")
    base_columns = [column for column in df.columns if column not in columns]
    anchor_index = base_columns.index(anchor_column)
    return df.loc[:, base_columns[:anchor_index] + columns + base_columns[anchor_index:]]


def _insert_columns_after(df, anchor_column, columns):
    if anchor_column not in df.columns:
        raise ValueError(f"Cannot insert after missing column: {anchor_column}")
    base_columns = [column for column in df.columns if column not in columns]
    anchor_index = base_columns.index(anchor_column) + 1
    return df.loc[:, base_columns[:anchor_index] + columns + base_columns[anchor_index:]]


def _graphormer_rt_tanaka_values_by_cc(raw_cc_path, cc_map_path, column_properties_path):
    from src.benchmarks.graphormer_rt.data.build_report_rp_inputs import (
        _load_column_properties,
        _tanaka_hsmb_values,
    )

    raw_cc = pd.read_csv(raw_cc_path, sep="\t", dtype=str).fillna("")
    cc_map = pd.read_csv(cc_map_path, sep="\t", dtype=str)
    properties_by_dir_id, properties_by_column_key = _load_column_properties(Path(column_properties_path))
    raw_by_dir_id = {str(row["dir_id"]).zfill(4): row for _, row in raw_cc.iterrows()}

    rows = []
    for _, cc_row in cc_map.iterrows():
        cc_id = cc_row["cc_id"]
        dir_ids = parse_dir_ids(cc_row["dir_ids"])
        candidate_values = []
        sources = []
        for dir_id in dir_ids:
            raw_row = raw_by_dir_id.get(dir_id)
            if raw_row is None:
                continue
            values, source = _tanaka_hsmb_values(
                dir_id,
                raw_row,
                properties_by_dir_id,
                properties_by_column_key,
                "zero",
            )
            tanaka_values = tuple(values[: len(TANAKA_VALUE_COLUMNS)])
            if has_nonzero_tanaka(tanaka_values):
                candidate_values.append(tanaka_values)
                sources.append(f"{dir_id}:{source}")

        unique_values = sorted(set(candidate_values))
        if len(unique_values) > 1:
            raise ValueError(
                f"Ambiguous Tanaka values for {cc_id}: {unique_values}. "
                "The article pipeline requires a unique vector for every condition."
            )

        output_row = {"cc_id": cc_id, "dir_ids": "|".join(dir_ids)}
        if unique_values:
            selected_values = unique_values[0]
            output_row.update(dict(zip(TANAKA_VALUE_COLUMNS, selected_values)))
            output_row["tanaka_source"] = "|".join(sorted(set(sources)))
        else:
            for column in TANAKA_VALUE_COLUMNS:
                output_row[column] = np.nan
            output_row["tanaka_source"] = ""
        rows.append(output_row)

    return pd.DataFrame(rows)


def _load_curated_tanaka_values(curated_tanaka_path):
    curated = pd.read_csv(curated_tanaka_path, sep="\t", dtype=str).fillna("")
    required_columns = ["column_key", *TANAKA_VALUE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in curated.columns]
    if missing_columns:
        raise ValueError(f"Curated Tanaka table is missing required columns: {missing_columns}")
    if (curated["column_key"].str.strip() == "").any():
        raise ValueError("Curated Tanaka table contains an empty column_key")
    if curated["column_key"].duplicated().any():
        raise ValueError("Curated Tanaka table contains duplicate column_key values")
    for column in TANAKA_VALUE_COLUMNS:
        curated[column] = pd.to_numeric(curated[column], errors="raise")
        if curated[column].isna().any():
            raise ValueError(f"Curated Tanaka table contains missing values in {column}")
    return curated


def _overlay_curated_tanaka_values(
    tanaka_by_cc,
    raw_cc_path,
    cc_map_path,
    curated_tanaka_path,
):
    """Fill missing vectors from manual curation by stationary-phase model."""
    raw_metadata = load_raw_column_metadata(raw_cc_path=raw_cc_path, cc_map_path=cc_map_path)
    curated = _load_curated_tanaka_values(curated_tanaka_path)
    matched = raw_metadata.loc[:, ["cc_id", "column_key"]].merge(
        curated,
        on="column_key",
        how="inner",
        validate="many_to_one",
    )

    output = tanaka_by_cc.copy().set_index("cc_id")
    for column in TANAKA_VALUE_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    for _, row in matched.iterrows():
        cc_id = row["cc_id"]
        if cc_id not in output.index:
            continue
        if output.loc[cc_id, TANAKA_VALUE_COLUMNS].notna().all():
            continue
        output.loc[cc_id, TANAKA_VALUE_COLUMNS] = row[TANAKA_VALUE_COLUMNS].to_numpy(dtype=float)
        source = str(row.get("tanaka_source", "article_curated")).strip() or "article_curated"
        output.loc[cc_id, "tanaka_source"] = source
    return output.reset_index()


def enrich_with_graphormer_rt_tanaka(
    processed_path,
    output_path,
    raw_cc_path,
    cc_map_path,
    column_properties_path,
    report_dir=None,
):
    processed_df = pd.read_csv(processed_path, sep="\t")
    tanaka_by_cc = _graphormer_rt_tanaka_values_by_cc(
        raw_cc_path=raw_cc_path,
        cc_map_path=cc_map_path,
        column_properties_path=column_properties_path,
    )
    tanaka_by_cc = _overlay_curated_tanaka_values(
        tanaka_by_cc=tanaka_by_cc,
        raw_cc_path=raw_cc_path,
        cc_map_path=cc_map_path,
        curated_tanaka_path=ARTICLE_CURATED_TANAKA_PATH,
    )

    enriched = processed_df.merge(
        tanaka_by_cc.loc[:, ["cc_id", *TANAKA_VALUE_COLUMNS]],
        on="cc_id",
        how="left",
    )
    mask_data = {}
    for value_column, mask_column in zip(TANAKA_VALUE_COLUMNS, TANAKA_MASK_COLUMNS):
        mask_data[mask_column] = enriched[value_column].isna().astype(np.float32)
        enriched[value_column] = enriched[value_column].fillna(0.0).astype(np.float32)
    enriched = pd.concat([enriched, pd.DataFrame(mask_data, index=enriched.index)], axis=1)

    enriched = _insert_columns_before(enriched, "column.length", TANAKA_MASK_COLUMNS)
    enriched = _insert_columns_after(enriched, "column.t0", TANAKA_VALUE_COLUMNS)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, sep="\t", index=False)

    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        coverage = tanaka_by_cc.copy()
        coverage["has_tanaka"] = coverage[TANAKA_VALUE_COLUMNS].notna().all(axis=1)
        coverage = coverage[coverage["cc_id"].isin(processed_df["cc_id"].unique())]
        coverage.to_csv(report_dir / "tanaka_graphormer_rt_coverage_report.tsv", sep="\t", index=False)
        coverage.loc[~coverage["has_tanaka"]].to_csv(
            report_dir / "tanaka_graphormer_rt_unmatched_cc.tsv",
            sep="\t",
            index=False,
        )
    return output_path
