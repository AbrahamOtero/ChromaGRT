"""Rebuild the ChromaGRT RepoRT RP data pipeline from revision 35b1bb689b310a2281f10efdcce223bb0007d74b.

Downloads RepoRT raw data at one Git commit, produces the curated
RP variants, adds the article's Tanaka features and molecular descriptors, and
optionally creates the shared assets from the versioned article fold manifests.
Generated files are under /data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.RepoRT_data_processing.RepoRT_get_raw_data import (
    merge_complete_file,
    repo_rt_processed_data_url,
)
from src.RepoRT_data_processing.RepoRT_preprocessing import preprocess_rp_dataset
from src.RepoRT_data_processing.RepoRT_processing import get_processed_df_from_raw
from src.benchmarks.graphormer_rt.data.extract_column_properties import extract_properties
from src.benchmarks.shared_folds.build_assets_from_fold_manifest import build_assets
from src.get_molecular_descriptors.get_molecular_descriptors import complete_moldesc_query
from src.training.RepoRT.tanaka_parameters import (
    ARTICLE_CURATED_TANAKA_PATH,
    enrich_with_graphormer_rt_tanaka,
)


ARTICLE_VARIANTS = ("no_SMRT", "with_SMRT", "with_SMRT_full")
ARTICLE_REPORTRT_REVISION = "35b1bb689b310a2281f10efdcce223bb0007d74b"
ARTICLE_REPO_IDS = list(range(1, 393))
ARTICLE_FOLD_MANIFEST_RELATIVE_PATH = Path("fold_manifests") / "article-v1"
SMRT_DIR_ID = "0186"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assert_absent(paths: list[Path], label: str) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite {label}: " + ", ".join(existing))


def _raw_paths(data_root: Path) -> list[Path]:
    raw_dir = data_root / "RepoRT" / "raw_data"
    rp_raw_dir = data_root / "RepoRT_RP" / "raw_data"
    return [
        raw_dir / "raw_rt_data.tsv",
        raw_dir / "raw_cc_data.tsv",
        raw_dir / "raw_grad_data.tsv",
        rp_raw_dir / "raw_rt_data.tsv",
        rp_raw_dir / "raw_cc_data.tsv",
        rp_raw_dir / "raw_grad_data.tsv",
    ]


def _raw_provenance_path(data_root: Path) -> Path:
    return data_root / "RepoRT" / "raw_data" / "repo_rt_source.json"


def _write_raw_provenance(data_root: Path, revision: str, repo_ids: list[int]) -> None:
    provenance_path = _raw_provenance_path(data_root)
    with provenance_path.open("w") as handle:
        json.dump(
            {
                "repo_rt_revision": revision,
                "repo_rt_processed_data_url": repo_rt_processed_data_url(revision),
                "repo_ids": repo_ids,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def _validate_existing_raw_provenance(data_root: Path, revision: str, repo_ids: list[int]) -> None:
    provenance_path = _raw_provenance_path(data_root)
    if not provenance_path.is_file():
        raise FileNotFoundError(
            "--skip-download requires " + str(provenance_path) + " to identify the raw RepoRT revision"
        )
    with provenance_path.open() as handle:
        provenance = json.load(handle)
    if provenance.get("repo_rt_revision") != revision or provenance.get("repo_ids") != repo_ids:
        raise ValueError("Existing raw RepoRT provenance does not match the requested revision")


def _variant_paths(data_root: Path, variant: str) -> tuple[Path, Path, Path]:
    rp_root = data_root / "RepoRT_RP"
    if variant == "with_SMRT_full":
        preprocessed_dir = rp_root / "preprocessed_data" / "with_SMRT_full"
    else:
        preprocessed_dir = rp_root / "preprocessed_data"
    processed_dir = rp_root / "processed_data" / variant
    artifacts_dir = rp_root / "processed_data" / f"{variant}_build_artifacts"
    return preprocessed_dir, processed_dir, artifacts_dir


def _prepare_variant(data_root: Path, variant: str, column_properties: Path) -> dict[str, object]:
    preprocessed_dir, processed_dir, artifacts_dir = _variant_paths(data_root, variant)
    rp_raw_root = data_root / "RepoRT_RP"

    if variant == "with_SMRT_full":
        _assert_absent([preprocessed_dir], f"{variant} preprocessing output")
        preprocess_rp_dataset(
            raw_root=rp_raw_root,
            output_dir=preprocessed_dir,
            filter_smrt_by_npls=False,
        )

    _assert_absent([processed_dir, artifacts_dir], f"{variant} processing output")
    get_processed_df_from_raw(
        source_path=artifacts_dir,
        drop_smrt=variant == "no_SMRT",
        down_grad_filter=False,
        smrt_dir_id=SMRT_DIR_ID if variant == "no_SMRT" else None,
        preprocessed_dir=preprocessed_dir,
        output_dir=processed_dir,
        raw_data_dir=rp_raw_root / "raw_data",
    )

    output_path = processed_dir / "complete_processed_data_with_tanaka.tsv"
    enrich_with_graphormer_rt_tanaka(
        processed_path=processed_dir / "complete_processed_data.tsv",
        output_path=output_path,
        raw_cc_path=rp_raw_root / "raw_data" / "raw_cc_data.tsv",
        cc_map_path=artifacts_dir / "cc_id_vs_dir_id.tsv",
        column_properties_path=column_properties,
        report_dir=processed_dir / "tanaka_reports",
    )
    complete_df = pd.read_csv(output_path, sep="\t", usecols=["cc_id", "molecule_id"])
    return {
        "preprocessed_dir": str(preprocessed_dir),
        "processed_dir": str(processed_dir),
        "rows": int(len(complete_df)),
        "cc_ids": int(complete_df["cc_id"].nunique()),
    }


def _build_article_assets(data_root: Path, manifest_dir: Path, column_properties: Path) -> dict[str, dict]:
    input_path = data_root / "RepoRT_RP" / "processed_data" / "no_SMRT" / "complete_processed_data_with_tanaka.tsv"
    preprocessed_dir = data_root / "RepoRT_RP" / "preprocessed_data"
    output_root = data_root / "benchmarks" / manifest_dir.name
    scenarios = {
        "random": "random_10fold.tsv",
        "bm_scaffold": "bm_scaffold_10fold.tsv",
        "cc": "cc_10fold.tsv",
    }
    _assert_absent([output_root / scenario for scenario in scenarios], "article fold assets")
    results = {}
    for scenario, filename in scenarios.items():
        results[scenario] = build_assets(
            SimpleNamespace(
                scenario=scenario,
                input_path=input_path,
                fold_manifest=manifest_dir / filename,
                output_dir=output_root / scenario,
                preprocessed_cc=preprocessed_dir / "preprocessed_cc_data.tsv",
                preprocessed_grad=preprocessed_dir / "preprocessed_gradient_data.tsv",
                column_properties=column_properties,
                molecule_id_col="molecule_id",
                smiles_col="smiles.std",
                rt_col="rt",
                cc_col="cc_id",
                scaffold_col="ms_smiles",
                n_folds=10,
                gradient_time_scale=1.0 / 60.0,
                missing_column_properties="zero",
            )
        )
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_project_root() / "data")
    parser.add_argument("--skip-download", action="store_true", help="Use an already downloaded raw RepoRT tree.")
    parser.add_argument("--skip-molecular-descriptors", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    raw_paths = _raw_paths(data_root)
    if args.skip_download:
        missing = [str(path) for path in raw_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing --skip-download raw inputs: " + ", ".join(missing))
        _validate_existing_raw_provenance(data_root, ARTICLE_REPORTRT_REVISION, ARTICLE_REPO_IDS)
    else:
        _assert_absent(raw_paths, "RepoRT raw data")
        merge_complete_file(
            path2res=data_root / "RepoRT" / "raw_data",
            seed_url=repo_rt_processed_data_url(ARTICLE_REPORTRT_REVISION),
            repos_array=ARTICLE_REPO_IDS,
            split_output_root=data_root,
        )
        _write_raw_provenance(data_root, ARTICLE_REPORTRT_REVISION, ARTICLE_REPO_IDS)

    column_properties = data_root / "benchmarks" / "graphormer_rt" / "column_properties_from_upstream_sample.tsv"
    _assert_absent([column_properties], "GraphormerRT column-properties table")
    extract_properties(
        metadata_pickle=_project_root() / "src" / "benchmarks" / "graphormer_rt" / "upstream" / "sample_data" / "RP_metadata.pickle",
        output_path=column_properties,
    )

    default_preprocessed, _, _ = _variant_paths(data_root, "no_SMRT")
    _assert_absent([default_preprocessed], "NPLS-filtered RP preprocessing output")
    preprocess_rp_dataset(
        raw_root=data_root / "RepoRT_RP",
        output_dir=default_preprocessed,
        filter_smrt_by_npls=True,
    )

    variants = {
        variant: _prepare_variant(data_root, variant, column_properties)
        for variant in ARTICLE_VARIANTS
    }

    descriptor_path = data_root / "complete_moldesc.tsv"
    if not args.skip_molecular_descriptors:
        _assert_absent([data_root / "moldescs.tsv", descriptor_path], "molecular-descriptor outputs")
        complete_moldesc_query(
            initial_file=data_root / "moldescs.tsv",
            path2res=descriptor_path,
            raw_report_file=data_root / "RepoRT" / "raw_data" / "raw_rt_data.tsv",
        )
    elif not descriptor_path.is_file():
        raise FileNotFoundError(
            "--skip-molecular-descriptors requires an existing compatible descriptor table: "
            + str(descriptor_path)
        )

    assets = _build_article_assets(
        data_root,
        _project_root() / ARTICLE_FOLD_MANIFEST_RELATIVE_PATH,
        column_properties,
    )

    manifest_path = data_root / "RepoRT_RP" / "reproduction_manifest.json"
    _assert_absent([manifest_path], "pipeline provenance manifest")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        json.dump(
            {
                "repo_rt_revision": ARTICLE_REPORTRT_REVISION,
                "repo_rt_processed_data_url": repo_rt_processed_data_url(ARTICLE_REPORTRT_REVISION),
                "repo_ids": ARTICLE_REPO_IDS,
                "variants": variants,
                "article_curated_tanaka": str(ARTICLE_CURATED_TANAKA_PATH),
                "molecular_descriptors": str(descriptor_path),
                "fold_manifest_dir": str(_project_root() / ARTICLE_FOLD_MANIFEST_RELATIVE_PATH),
                "article_assets": assets,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print(f"Pipeline provenance: {manifest_path}")


if __name__ == "__main__":
    main()
