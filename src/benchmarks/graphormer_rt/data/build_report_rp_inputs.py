"""Build GraphormerRT-compatible RP inputs from local RepoRT files.

The GraphormerRT RP loader expects:

- A headerless CSV with "method_id,SMILES,rt_minutes" rows;
- "RP_metadata.pickle" containing "method_id -> 100 metadata fields".

This script reconstructs that structure from this project's local RepoRT RP
raw/preprocessed files without depending on the current project model code.
Fields that are not available locally, such as Tanaka/HSMB parameters, are
recovered from the local column-properties table when possible and otherwise
set to 0 so the upstream featurizer can still run.  The builder implements
only the RP reconstruction used for the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path
from typing import Any

import pandas as pd


METADATA_HEADER = [
    "company_name",
    "usp_code",
    "col_length",
    "col_innerdiam",
    "col_part_size",
    "temp",
    "col_fl",
    "col_dead",
    "HPLC_type",
    "A_solv",
    "B_solv",
    "time1",
    "grad1",
    "time2",
    "grad2",
    "time3",
    "grad3",
    "time4",
    "grad4",
    "A_pH",
    "B_pH",
    "A_start",
    "A_end",
    "B_start",
    "B_end",
    "eluent.A.formic",
    "eluent.A.formic.unit",
    "eluent.A.acetic",
    "eluent.A.acetic.unit",
    "eluent.A.trifluoroacetic",
    "eluent.A.trifluoroacetic.unit",
    "eluent.A.phosphor",
    "eluent.A.phosphor.unit",
    "eluent.A.nh4ac",
    "eluent.A.nh4ac.unit",
    "eluent.A.nh4form",
    "eluent.A.nh4form.unit",
    "eluent.A.nh4carb",
    "eluent.A.nh4carb.unit",
    "eluent.A.nh4bicarb",
    "eluent.A.nh4bicarb.unit",
    "eluent.A.nh4f",
    "eluent.A.nh4f.unit",
    "eluent.A.nh4oh",
    "eluent.A.nh4oh.unit",
    "eluent.A.trieth",
    "eluent.A.trieth.unit",
    "eluent.A.triprop",
    "eluent.A.triprop.unit",
    "eluent.A.tribut",
    "eluent.A.tribut.unit",
    "eluent.A.nndimethylhex",
    "eluent.A.nndimethylhex.unit",
    "eluent.A.medronic",
    "eluent.A.medronic.unit",
    "eluent.B.formic",
    "eluent.B.formic.unit",
    "eluent.B.acetic",
    "eluent.B.acetic.unit",
    "eluent.B.trifluoroacetic",
    "eluent.B.trifluoroacetic.unit",
    "eluent.B.phosphor",
    "eluent.B.phosphor.unit",
    "eluent.B.nh4ac",
    "eluent.B.nh4ac.unit",
    "eluent.B.nh4form",
    "eluent.B.nh4form.unit",
    "eluent.B.nh4carb",
    "eluent.B.nh4carb.unit",
    "eluent.B.nh4bicarb",
    "eluent.B.nh4bicarb.unit",
    "eluent.B.nh4f",
    "eluent.B.nh4f.unit",
    "eluent.B.nh4oh",
    "eluent.B.nh4oh.unit",
    "eluent.B.trieth",
    "eluent.B.trieth.unit",
    "eluent.B.triprop",
    "eluent.B.triprop.unit",
    "eluent.B.tribut",
    "eluent.B.tribut.unit",
    "eluent.B.nndimethylhex",
    "eluent.B.nndimethylhex.unit",
    "eluent.B.medronic",
    "eluent.B.medronic.unit",
    "kPB",
    "alphaCH2",
    "alphaT/O",
    "alphaC/P",
    "alphaB/P",
    "alphaB/P.1",
    "particle size",
    "pore size",
    "H",
    "S*",
    "A",
    "B",
    "C (pH 2.8)",
    "C (pH 7.0)",
    "EB retention factor",
]

COMPANIES = ["Waters", "Thermo", "Agilent", "Restek", "Merck", "Phenomenex", "HILICON"]
USPS = ["L1", "L10", "L109", "L11", "L43", "L68", "L3", "L114", "L112", "L122"]
SOLVENTS = ["h2o", "meoh", "acn"]
MOBILE_PHASE_SOLVENTS = ["h2o", "meoh", "acn", "iproh", "acetone", "hex", "chcl3", "ch2cl2", "hept"]
DISALLOWED_RP_SOLVENTS = ["iproh", "acetone"]
PHASES = ["A", "B", "C", "D"]
ADDITIVES = [
    "formic",
    "acetic",
    "trifluoroacetic",
    "phosphor",
    "nh4ac",
    "nh4form",
    "nh4carb",
    "nh4bicarb",
    "nh4f",
    "nh4oh",
    "trieth",
    "triprop",
    "tribut",
    "nndimethylhex",
    "medronic",
]
TANAKA_HSMB_FIELDS = METADATA_HEADER[85:]
ARTICLE_MAX_DIR_ID = 393


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def _value(row: pd.Series, column: str, default: Any = "0") -> Any:
    value = row.get(column, default)
    if value == "" or pd.isna(value):
        return default
    return value


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str_number(value: Any, default: str = "0") -> str:
    number = _to_float(value, _to_float(default, 0.0))
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _company(column_name: str) -> str:
    lowered = str(column_name).lower()
    for company in COMPANIES:
        if company.lower() in lowered:
            return company
    return "Other"


def _solvent(row: pd.Series, phase: str) -> str:
    values = {solvent: _to_float(_value(row, f"eluent.{phase}.{solvent}")) for solvent in SOLVENTS}
    solvent, amount = max(values.items(), key=lambda item: item[1])
    if amount > 0:
        return solvent
    return "Other"


def _phase_has_solvent(row: pd.Series, phase: str) -> bool:
    return any(_to_float(_value(row, f"eluent.{phase}.{solvent}")) > 0 for solvent in MOBILE_PHASE_SOLVENTS)


def _gradient_phase_used(row: pd.Series, phase: str) -> bool:
    for idx in range(18):
        if _value(row, f"t [min]_{idx}", "") == "":
            continue
        if _to_float(_value(row, f"{phase} [%]_{idx}")) > 0:
            return True
    return False


def _has_disallowed_rp_solvent(row: pd.Series) -> bool:
    return any(
        _to_float(_value(row, f"eluent.{phase}.{solvent}")) > 0
        for phase in PHASES
        for solvent in DISALLOWED_RP_SOLVENTS
    )


def _mobile_phase_count(cc_row: pd.Series, grad_row: pd.Series) -> int:
    return sum(
        1
        for phase in PHASES
        if _phase_has_solvent(cc_row, phase) or _gradient_phase_used(grad_row, phase)
    )


def _distinct_mobile_solvent_count(cc_row: pd.Series) -> int:
    solvents = set()
    for phase in PHASES:
        for solvent in MOBILE_PHASE_SOLVENTS:
            if _to_float(_value(cc_row, f"eluent.{phase}.{solvent}")) > 0:
                solvents.add(solvent)
    return len(solvents)


def _flow_rates(row: pd.Series) -> list[float]:
    values = []
    for idx in range(18):
        value = _value(row, f"flow rate [ml/min]_{idx}", "")
        if value != "":
            values.append(_to_float(value))
    return values


def _has_nonconstant_flow_rate(row: pd.Series) -> bool:
    values = _flow_rates(row)
    if len(values) <= 1:
        return False
    return max(values) - min(values) > 1e-9


def _paper_method_filter_reasons(cc_row: pd.Series, grad_row: pd.Series) -> list[str]:
    reasons = []
    if _has_disallowed_rp_solvent(cc_row):
        reasons.append("ipa_or_acetone")
    if _mobile_phase_count(cc_row, grad_row) > 2 or _distinct_mobile_solvent_count(cc_row) > 2:
        reasons.append("more_than_two_mobile_phases")
    if _has_nonconstant_flow_rate(grad_row):
        reasons.append("nonconstant_flow")
    if _to_float(_value(cc_row, "column.t0")) > 3:
        reasons.append("t0_gt_3_min")
    return reasons


def _gradient_inflection_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 2:
        return []
    max_index = max(range(len(points)), key=lambda idx: points[idx][1])
    inflections = []
    for idx in range(1, len(points)):
        if points[idx][1] != points[idx - 1][1] and idx <= max_index:
            inflections.extend([points[idx - 1], points[idx]])

    filtered = []
    previous_time = -1.0
    for point in sorted(set(inflections), key=lambda item: item[0]):
        if abs(point[0] - previous_time) < 0.3:
            continue
        filtered.append(point)
        previous_time = point[0]
    return filtered


def _upstream_extra_gradient_filter_reasons(grad_row: pd.Series) -> list[str]:
    points = _gradient_points(grad_row, organic_phase=_organic_phase_from_gradient(grad_row))
    reasons = []
    if points and max(point[1] for point in points) < 80:
        reasons.append("pB_max_lt_80")
    if len(_gradient_inflection_points(points)) > 3:
        reasons.append("more_than_three_inflection_points")
    return reasons


def _organic_phase_from_gradient(row: pd.Series) -> str:
    a0 = _to_float(_value(row, "A [%]_0"))
    b0 = _to_float(_value(row, "B [%]_0"))
    return "A" if b0 > 50 and a0 < b0 else "B"


def _usp_code(value: Any) -> str:
    value = str(value)
    return value if value in USPS else "Other"


def _method_id_int(value: str) -> int | None:
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    return int(match.group(0))


def _column_property_key(company: Any, usp: Any, length: Any, innerdiam: Any, particle_size: Any) -> tuple[str, str, str, str, str]:
    return (
        str(company or "Other"),
        _usp_code(usp),
        _as_str_number(length),
        _as_str_number(innerdiam),
        _as_str_number(particle_size),
    )


def _column_property_key_from_cc_row(cc_row: pd.Series) -> tuple[str, str, str, str, str]:
    return _column_property_key(
        _company(_value(cc_row, "column.name", "")),
        _value(cc_row, "column.usp.code", "Other"),
        _value(cc_row, "column.length"),
        _value(cc_row, "column.id"),
        _value(cc_row, "column.particle.size"),
    )


def _load_column_properties(path: Path | None) -> tuple[dict[str, dict[str, str]], dict[tuple[str, str, str, str, str], dict[str, str]]]:
    if path is None or not path.exists():
        return {}, {}
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    by_dir_id: dict[str, dict[str, str]] = {}
    by_column_key: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for _, row in df.iterrows():
        values = {field: _as_str_number(row.get(field, "0")) for field in TANAKA_HSMB_FIELDS}
        dir_id = str(row.get("dir_id", "")).strip()
        if dir_id:
            by_dir_id[_normalise_method_id(dir_id)] = values
        key = _column_property_key(
            row.get("company_name", "Other"),
            row.get("usp_code", "Other"),
            row.get("col_length", "0"),
            row.get("col_innerdiam", "0"),
            row.get("col_part_size", "0"),
        )
        by_column_key.setdefault(key, values)
    return by_dir_id, by_column_key


def _tanaka_hsmb_values(
    method_id: str,
    cc_row: pd.Series,
    properties_by_dir_id: dict[str, dict[str, str]],
    properties_by_column_key: dict[tuple[str, str, str, str, str], dict[str, str]],
    missing_policy: str,
) -> tuple[list[str], str]:
    if method_id in properties_by_dir_id:
        properties = properties_by_dir_id[method_id]
        return [properties.get(field, "0") for field in TANAKA_HSMB_FIELDS], "dir_id"

    key = _column_property_key_from_cc_row(cc_row)
    if key in properties_by_column_key:
        properties = properties_by_column_key[key]
        return [properties.get(field, "0") for field in TANAKA_HSMB_FIELDS], "column_key"

    if missing_policy == "error":
        raise ValueError(f"Missing Tanaka/HSMB column properties for method {method_id} and key {key}")
    return ["0"] * len(TANAKA_HSMB_FIELDS), "missing_zero"


def _gradient_points(row: pd.Series, organic_phase: str = "B") -> list[tuple[float, float]]:
    points = []
    for idx in range(18):
        t = _value(row, f"t [min]_{idx}", "")
        b = _value(row, f"{organic_phase} [%]_{idx}", "")
        if t == "" or b == "":
            continue
        points.append((_to_float(t), _to_float(b)))
    return points


def _gradient_value(points: list[tuple[float, float]], idx: int, item: int) -> str:
    if idx >= len(points):
        return "0"
    return _as_str_number(points[idx][item])


def _metadata_for_method(
    method_id: str,
    cc_row: pd.Series,
    grad_row: pd.Series,
    properties_by_dir_id: dict[str, dict[str, str]],
    properties_by_column_key: dict[tuple[str, str, str, str, str], dict[str, str]],
    missing_column_properties: str,
) -> tuple[list[Any], str]:
    organic_phase = _organic_phase_from_gradient(grad_row)
    aqueous_phase = "B" if organic_phase == "A" else "A"
    points = _gradient_points(grad_row, organic_phase=organic_phase)
    b_start = _to_float(_gradient_value(points, 0, 1))
    b_end = _to_float(_gradient_value(points, len(points) - 1, 1)) if points else b_start
    values: list[Any] = [
        _company(_value(cc_row, "column.name", "")),
        _usp_code(_value(cc_row, "column.usp.code", "Other")),
        _as_str_number(_value(cc_row, "column.length")),
        _as_str_number(_value(cc_row, "column.id")),
        _as_str_number(_value(cc_row, "column.particle.size")),
        _as_str_number(_value(cc_row, "column.temperature")),
        _to_float(_value(cc_row, "column.flowrate")),
        _as_str_number(_value(cc_row, "column.t0")),
        "RP",
        _solvent(cc_row, aqueous_phase),
        _solvent(cc_row, organic_phase),
        _gradient_value(points, 0, 0),
        _gradient_value(points, 0, 1),
        _gradient_value(points, 1, 0),
        _gradient_value(points, 1, 1),
        _gradient_value(points, 2, 0),
        _gradient_value(points, 2, 1),
        _gradient_value(points, 3, 0),
        _gradient_value(points, 3, 1),
        _as_str_number(_value(cc_row, f"eluent.{aqueous_phase}.pH", "3")),
        _as_str_number(_value(cc_row, f"eluent.{organic_phase}.pH", "3")),
        _as_str_number(100 - b_start),
        _as_str_number(100 - b_end),
        _as_str_number(b_start),
        _as_str_number(b_end),
    ]

    for phase in [aqueous_phase, organic_phase]:
        for additive in ADDITIVES:
            values.append(_as_str_number(_value(cc_row, f"eluent.{phase}.{additive}")))
            values.append(_value(cc_row, f"eluent.{phase}.{additive}.unit", ""))

    tanaka_hsmb_values, property_source = _tanaka_hsmb_values(
        method_id,
        cc_row,
        properties_by_dir_id,
        properties_by_column_key,
        missing_column_properties,
    )
    values.extend(tanaka_hsmb_values)
    if len(values) != len(METADATA_HEADER):
        raise RuntimeError(f"Unexpected metadata length: {len(values)}")
    return values, property_source


def _normalise_method_id(value: str) -> str:
    match = re.search(r"\d+", str(value))
    if not match:
        return str(value)
    return match.group(0).zfill(4)


def build_inputs(
    raw_dir: Path,
    preprocessed_dir: Path,
    output_dir: Path,
    column_properties_path: Path,
) -> dict[str, Any]:
    rt_df = _read_tsv(raw_dir / "raw_rt_data.tsv")
    cc_df = _read_tsv(preprocessed_dir / "preprocessed_cc_data.tsv")
    grad_df = _read_tsv(preprocessed_dir / "preprocessed_gradient_data.tsv")

    cc_df["dir_id"] = cc_df["dir_id"].map(_normalise_method_id)
    grad_df["dir_id"] = grad_df["dir_id"].map(_normalise_method_id)
    rt_df["dir_id"] = rt_df["dir_id"].map(_normalise_method_id)

    cc_df = cc_df[cc_df["dir_id"].map(lambda value: (_method_id_int(value) or 0) <= ARTICLE_MAX_DIR_ID)]
    grad_df = grad_df[grad_df["dir_id"].map(lambda value: (_method_id_int(value) or 0) <= ARTICLE_MAX_DIR_ID)]
    rt_df = rt_df[rt_df["dir_id"].map(lambda value: (_method_id_int(value) or 0) <= ARTICLE_MAX_DIR_ID)]

    cc_by_id = {row["dir_id"]: row for _, row in cc_df.iterrows()}
    grad_by_id = {row["dir_id"]: row for _, row in grad_df.iterrows()}
    initial_method_ids = sorted(set(cc_by_id) & set(grad_by_id))
    method_filter_reasons: dict[str, list[str]] = {}
    method_ids = []
    for method_id in initial_method_ids:
        reasons = _paper_method_filter_reasons(cc_by_id[method_id], grad_by_id[method_id])
        reasons.extend(_upstream_extra_gradient_filter_reasons(grad_by_id[method_id]))
        if reasons:
            method_filter_reasons[method_id] = reasons
        else:
            method_ids.append(method_id)
    properties_by_dir_id, properties_by_column_key = _load_column_properties(column_properties_path)

    metadata: dict[str, list[Any]] = {"1": METADATA_HEADER}
    missing_metadata = sorted((set(cc_by_id) | set(grad_by_id)) - set(method_ids))
    property_source_counts = {"dir_id": 0, "column_key": 0, "missing_zero": 0}
    property_source_methods = {"dir_id": [], "column_key": [], "missing_zero": []}
    for method_id in method_ids:
        metadata[method_id], property_source = _metadata_for_method(
            method_id,
            cc_by_id[method_id],
            grad_by_id[method_id],
            properties_by_dir_id,
            properties_by_column_key,
            "zero",
        )
        property_source_counts[property_source] = property_source_counts.get(property_source, 0) + 1
        property_source_methods.setdefault(property_source, []).append(method_id)

    usable_rt = rt_df[
        rt_df["dir_id"].isin(method_ids)
        & rt_df["smiles.std"].astype(bool)
        & rt_df["rt"].astype(bool)
    ].copy()
    usable_rt["rt_numeric"] = usable_rt["rt"].map(_to_float)
    initial_usable_rt_rows = int(len(usable_rt))
    t0_by_method = {method_id: _to_float(_value(cc_by_id[method_id], "column.t0")) for method_id in method_ids}
    t0_values = usable_rt["dir_id"].map(t0_by_method)
    retained_mask = usable_rt["rt_numeric"] > t0_values
    non_retained_rows_removed = int((~retained_mask).sum())
    usable_rt = usable_rt[retained_mask].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "report_rp.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for _, row in usable_rt.iterrows():
            writer.writerow([row["dir_id"], row["smiles.std"], _as_str_number(row["rt"])])

    metadata_path = output_dir / "RP_metadata.pickle"
    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)

    summary = {
        "dataset_version": "paper_reproduction",
        "max_dir_id": ARTICLE_MAX_DIR_ID,
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path),
        "column_properties_path": str(column_properties_path),
        "paper_method_filters_applied": True,
        "upstream_extra_gradient_filters_applied": True,
        "non_retained_filter": "rt_le_t0",
        "n_initial_methods_with_metadata": int(len(initial_method_ids)),
        "n_methods_removed_by_paper_filters": int(len(method_filter_reasons)),
        "method_filter_reason_counts": {
            reason: sum(reason in reasons for reasons in method_filter_reasons.values())
            for reason in [
                "ipa_or_acetone",
                "more_than_two_mobile_phases",
                "nonconstant_flow",
                "t0_gt_3_min",
                "pB_max_lt_80",
                "more_than_three_inflection_points",
            ]
        },
        "method_filter_removed_methods": method_filter_reasons,
        "n_rt_rows_before_non_retained_filter": initial_usable_rt_rows,
        "n_non_retained_rows_removed": non_retained_rows_removed,
        "n_csv_rows": int(len(usable_rt)),
        "n_metadata_methods": int(len(metadata) - 1),
        "n_missing_metadata_methods": int(len(missing_metadata)),
        "missing_metadata_methods": missing_metadata[:50],
        "metadata_fields": len(METADATA_HEADER),
        "tanaka_hsmb_fields": TANAKA_HSMB_FIELDS,
        "tanaka_hsmb_property_source_counts": property_source_counts,
        "tanaka_hsmb_missing_zero_methods": property_source_methods.get("missing_zero", [])[:100],
        "tanaka_hsmb_fields_filled_with_zero": property_source_counts.get("missing_zero", 0) > 0,
    }
    with (output_dir / "build_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/RepoRT_RP/raw_data")
    parser.add_argument("--preprocessed-dir", default="data/RepoRT_RP/preprocessed_data")
    parser.add_argument("--column-properties", default="data/benchmarks/graphormer_rt/column_properties_from_upstream_sample.tsv")
    parser.add_argument("--output-dir", default="data/benchmarks/graphormer_rt/report_rp_paper")
    args = parser.parse_args()

    summary = build_inputs(
        raw_dir=Path(args.raw_dir),
        preprocessed_dir=Path(args.preprocessed_dir),
        output_dir=Path(args.output_dir),
        column_properties_path=Path(args.column_properties),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
