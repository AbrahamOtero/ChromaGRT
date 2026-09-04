"""Train and evaluate ChromaGRT on a predefined data split."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path

import pandas as pd
from lightning import pytorch as pl

from src.training.functions.splitted_sets_functions import (
    add_moldescs,
    get_deterministic_input_data,
    get_res_table,
    get_scaled_datasets,
    get_scaled_input_train_data,
    get_scaled_moldesc_testval,
    get_scaled_moldescs_train,
    metrics_from_dataframe,
    write_metric_txt,
    write_metrics_per_cc,
    write_parameters_file,
)
from src.training.model_backends import predict_chromagrt, train_chromagrt
from src.training.RepoRT.predefined_split.config import model_config_from_args, parse_args


def _input_paths(args):
    if args.shared_split_dir is None:
        return None, None
    return (
        args.shared_split_dir / "complete_manifest_dataset.tsv",
        args.shared_split_dir,
    )


def _results_path(args, timestamp):
    if args.results_dir is not None:
        results_base = args.results_dir
    else:
        descriptor_dir = "RepoRT_moldesc" if args.use_molecular_descriptors else "RepoRT_RP"
        results_base = (
            Path("logs")
            / descriptor_dir
            / args.dataset_type
            / "shared_folds"
            / "chromagrt"
            / args.run_name
        )
    return results_base / timestamp


def _print_configuration(args, model_config, input_file, split_path, results_path):
    runtime = {
        "dataset_type": args.dataset_type,
        "input_file": input_file,
        "molecular_descriptors_path": args.molecular_descriptors_path,
        "results_path": results_path,
        "run_name": args.run_name,
        "shared_split_dir": args.shared_split_dir,
        "split_path": split_path,
        "use_molecular_descriptors": args.use_molecular_descriptors,
    }
    print(json.dumps({"model": model_config, "runtime": runtime}, indent=2, default=str, sort_keys=True))


def main(argv=None):
    args = parse_args(argv)
    model_config = model_config_from_args(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file, split_path = _input_paths(args)
    results_path = _results_path(args, timestamp)

    if args.print_config:
        _print_configuration(args, model_config, input_file, split_path, results_path)
        return 0

    if split_path is None:
        raise ValueError(
            "A predefined split is required. Pass --shared-split-dir or set "
            "REPORT_SHARED_SPLIT_DIR."
        )

    pl.seed_everything(model_config["seed"], workers=True)
    train_file = split_path / "train_data.tsv"
    test_file = split_path / "test_data.tsv"
    val_file = split_path / "val_data.tsv"

    missing = [path for path in (train_file, val_file, test_file) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Shared split directory is missing files: {missing}")
    print(f"Using predefined split files from {split_path}")

    print("Reading input files.")
    train_df = pd.read_csv(train_file, sep="\t")
    test_df = pd.read_csv(test_file, sep="\t")
    val_df = pd.read_csv(val_file, sep="\t")
    results_path.mkdir(parents=True, exist_ok=True)
    results_path_string = str(results_path) + os.sep

    if args.use_molecular_descriptors:
        print("Adding and scaling molecular descriptors from the training split.")
        descriptor_path = str(args.molecular_descriptors_path)
        train_df = add_moldescs(train_df, descriptor_path)
        test_df = add_moldescs(test_df, descriptor_path)
        val_df = add_moldescs(val_df, descriptor_path)
        scaled_train_df, molecular_scaler = get_scaled_moldescs_train(train_df)
        scaled_test_df = get_scaled_moldesc_testval(test_df, molecular_scaler)
        scaled_val_df = get_scaled_moldesc_testval(val_df, molecular_scaler)
    else:
        scaled_train_df = train_df
        scaled_val_df = val_df
        scaled_test_df = test_df

    normalization = model_config["condition_normalization"]
    if normalization == "standard":
        print("Standardizing condition inputs with statistics from training conditions.")
        scaled_train_df, condition_scaler = get_scaled_input_train_data(scaled_train_df)
        scaled_val_df = get_scaled_datasets(scaled_val_df, condition_scaler)
        scaled_test_df = get_scaled_datasets(scaled_test_df, condition_scaler)
    elif normalization == "deterministic":
        print("Applying deterministic condition normalization.")
        scaled_train_df = get_deterministic_input_data(scaled_train_df)
        scaled_val_df = get_deterministic_input_data(scaled_val_df)
        scaled_test_df = get_deterministic_input_data(scaled_test_df)
    else:  # pragma: no cover - argparse constrains this value.
        raise ValueError(f"Unsupported condition normalization: {normalization}")

    run_config = {
        **model_config,
        "metadata_run_timestamp": timestamp,
        "metadata_model_run_name": args.run_name,
        "metadata_results_path": results_path_string,
        "metadata_shared_split_dir": str(args.shared_split_dir) if args.shared_split_dir else None,
        "metadata_split_source": "predefined",
        "metadata_dataset_type": args.dataset_type,
        "metadata_using_moldescs": args.use_molecular_descriptors,
        "metadata_model_backend": "chromagrt",
    }

    print("Building and training ChromaGRT.")
    model_bundle = train_chromagrt(
        scaled_train_df,
        scaled_val_df,
        run_config,
        results_path_string,
        using_moldescs=args.use_molecular_descriptors,
        save_model=True,
    )

    print(f"Writing evaluation outputs to {results_path}.")
    test_predictions = predict_chromagrt(model_bundle, scaled_test_df)
    results = get_res_table(
        test_df,
        test_predictions,
        results_path_string,
        using_moldescs=args.use_molecular_descriptors,
    )
    mae, rmse, mre, relative_max, relative_mean = metrics_from_dataframe(results)
    write_parameters_file(run_config, results_path_string)
    write_metrics_per_cc(results, results_path_string)
    write_metric_txt(mae, rmse, mre, relative_max, relative_mean, results_path_string)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
