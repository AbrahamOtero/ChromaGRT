"""Extract Tanaka/HSMB column properties from an metadata pickle."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

from src.benchmarks.graphormer_rt.data.build_report_rp_inputs import (
    METADATA_HEADER,
    TANAKA_HSMB_FIELDS,
    _as_str_number,
    _normalise_method_id,
)


def extract_properties(metadata_pickle: Path, output_path: Path) -> dict[str, int]:
    with metadata_pickle.open("rb") as handle:
        metadata = pickle.load(handle)

    fieldnames = ["dir_id", "company_name", "usp_code", "col_length", "col_innerdiam", "col_part_size", *TANAKA_HSMB_FIELDS]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    unique_keys = set()
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for method_id, values in sorted(metadata.items()):
            if method_id == "1":
                continue
            row = dict(zip(METADATA_HEADER, values))
            out = {
                "dir_id": _normalise_method_id(method_id),
                "company_name": row.get("company_name", "Other"),
                "usp_code": row.get("usp_code", "Other"),
                "col_length": _as_str_number(row.get("col_length", "0")),
                "col_innerdiam": _as_str_number(row.get("col_innerdiam", "0")),
                "col_part_size": _as_str_number(row.get("col_part_size", "0")),
            }
            for field in TANAKA_HSMB_FIELDS:
                out[field] = _as_str_number(row.get(field, "0"))
            writer.writerow(out)
            rows_written += 1
            unique_keys.add(tuple(out[field] for field in ["company_name", "usp_code", "col_length", "col_innerdiam", "col_part_size"]))

    return {"rows_written": rows_written, "unique_column_keys": len(unique_keys)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-pickle",
        default="src/benchmarks/graphormer_rt/upstream/sample_data/RP_metadata.pickle",
    )
    parser.add_argument(
        "--output",
        default="data/benchmarks/graphormer_rt/column_properties_from_upstream_sample.tsv",
    )
    args = parser.parse_args()

    summary = extract_properties(Path(args.metadata_pickle), Path(args.output))
    print(summary)


if __name__ == "__main__":
    main()
