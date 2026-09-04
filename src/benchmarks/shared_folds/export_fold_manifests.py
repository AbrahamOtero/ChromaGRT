"""Export compact, versionable assignments for the article cross-validation folds.

Not used by the article-reproduction pipeline. Retained to generate
new fold assignments if RepoRT dataset is refreshed.

The source "master_manifest.csv" files contain all model inputs.  
Keeps only the stable observation key and its base-fold assignment, so the
folds can be version separately from the dataset archive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ("cc_id", "molecule_id", "inchi.std")
SOURCE_COLUMNS = (*KEY_COLUMNS, "fold")
SCENARIOS = ("random", "bm_scaffold", "cc")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_source_manifest(path: Path, scenario: str, n_folds: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Source manifest does not exist: {path}")

    header = pd.read_csv(path, nrows=0)
    missing = set(SOURCE_COLUMNS) - set(header.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    rows = pd.read_csv(path, usecols=list(SOURCE_COLUMNS), dtype=str).fillna("")
    if rows.empty:
        raise ValueError(f"{path} contains no observations")
    if (rows.loc[:, KEY_COLUMNS] == "").any(axis=None):
        raise ValueError(f"{path} contains an empty observation key")
    if rows.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{path} contains duplicate observation keys")

    numeric_folds = pd.to_numeric(rows["fold"], errors="raise")
    if not (numeric_folds % 1 == 0).all():
        raise ValueError(f"{path} contains a non-integer fold assignment")
    rows["base_fold"] = numeric_folds.astype(int)
    rows = rows.drop(columns="fold")

    expected_folds = set(range(n_folds))
    actual_folds = set(rows["base_fold"])
    allowed_folds = expected_folds | ({-1} if scenario == "bm_scaffold" else set())
    invalid_folds = actual_folds - allowed_folds
    if invalid_folds:
        raise ValueError(f"{path} has unsupported fold assignments: {sorted(invalid_folds)}")
    missing_folds = expected_folds - actual_folds
    if missing_folds:
        raise ValueError(f"{path} is missing folds: {sorted(missing_folds)}")
    if scenario == "bm_scaffold" and -1 not in actual_folds:
        raise ValueError(f"{path} is missing the always-training BM fold (-1)")

    return rows.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)


def export_manifests(
    *,
    sources: dict[str, Path],
    output_dir: Path,
    dataset_label: str,
    dataset_version: str,
    n_folds: int,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")

    output_dir.mkdir(parents=True)
    scenario_metadata = {}
    try:
        for scenario in SCENARIOS:
            rows = _read_source_manifest(sources[scenario], scenario, n_folds)
            output_path = output_dir / f"{scenario}_10fold.tsv"
            rows.to_csv(output_path, sep="\t", index=False)
            scenario_metadata[scenario] = {
                "filename": output_path.name,
                "n_observations": int(len(rows)),
                "base_fold_counts": {
                    str(fold): int(count)
                    for fold, count in rows["base_fold"].value_counts().sort_index().items()
                },
            }
    except Exception:
        for generated_path in output_dir.iterdir():
            generated_path.unlink()
        output_dir.rmdir()
        raise

    metadata = {
        "dataset_label": dataset_label,
        "dataset_version": dataset_version,
        "key_columns": list(KEY_COLUMNS),
        "n_folds": n_folds,
        "split_policy": {
            "test": "base_fold == run index",
            "validation": "base_fold == (run index + 1) mod n_folds",
            "training": "all remaining observations",
            "bm_scaffold_base_fold_-1": "always training",
        },
        "scenarios": scenario_metadata,
    }
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-master", required=True, type=Path)
    parser.add_argument("--bm-master", required=True, type=Path)
    parser.add_argument("--cc-master", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        default=_project_root() / "fold_manifests" / "article-v1",
        type=Path,
    )
    parser.add_argument(
        "--dataset-label",
        default="Curated RepoRT reversed-phase no-SMRT dataset with Tanaka descriptors",
    )
    parser.add_argument("--dataset-version", default="article-v1")
    parser.add_argument("--n-folds", default=10, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = export_manifests(
        sources={
            "random": args.random_master,
            "bm_scaffold": args.bm_master,
            "cc": args.cc_master,
        },
        output_dir=args.output_dir,
        dataset_label=args.dataset_label,
        dataset_version=args.dataset_version,
        n_folds=args.n_folds,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
