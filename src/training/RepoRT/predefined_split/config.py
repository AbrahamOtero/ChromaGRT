"""Command-line configuration for ChromaGRT training runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.training.model_backends.chromagrt_defaults import DEFAULT_CHROMAGRT_CONFIG

PAPER_CHROMAGRT_DEFAULTS = DEFAULT_CHROMAGRT_CONFIG


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _csv_values(value: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return value
    if value.lower() in {"", "none", "null"}:
        return ()
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    return default if value is None else _csv_values(value)


def build_parser() -> argparse.ArgumentParser:
    defaults = PAPER_CHROMAGRT_DEFAULTS
    parser = argparse.ArgumentParser(
        description="Train and evaluate ChromaGRT on one predefined split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    runtime = parser.add_argument_group("data and output")
    runtime.add_argument(
        "--shared-split-dir",
        default=os.environ.get("REPORT_SHARED_SPLIT_DIR"),
        type=Path,
        help="Directory containing train_data.tsv, val_data.tsv and test_data.tsv.",
    )
    runtime.add_argument(
        "--results-dir",
        default=os.environ.get("REPORT_RESULTS_DIR"),
        type=Path,
        help="Base output directory; a timestamped directory is created inside it.",
    )
    runtime.add_argument(
        "--molecular-descriptors-path",
        default=Path(os.environ.get("REPORT_MOLDESC_PATH", "data/complete_moldesc.tsv")),
        type=Path,
    )
    runtime.add_argument(
        "--dataset-type",
        choices=["no_SMRT", "with_SMRT", "with_SMRT_full"],
        default=os.environ.get("REPORT_DATASET_TYPE", "no_SMRT"),
    )
    runtime.add_argument(
        "--use-molecular-descriptors",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("REPORT_USE_MOLDESCS", True),
    )
    runtime.add_argument(
        "--run-name",
        default=os.environ.get("CHROMAGRT_RUN_NAME", "paper_standard_tanaka"),
    )
    runtime.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved runtime/model configuration and exit without training.",
    )

    model = parser.add_argument_group("model inputs")
    model.add_argument("--batch-size", type=int, default=_env_int("CHROMAGRT_BATCH_SIZE", defaults["batch_size"]))
    model.add_argument("--num-workers", type=int, default=_env_int("CHROMAGRT_NUM_WORKERS", defaults["num_workers"]))
    model.add_argument(
        "--condition-normalization",
        choices=["standard", "deterministic"],
        default=os.environ.get("CHROMAGRT_CONDITION_NORMALIZATION", defaults["condition_normalization"]),
    )
    model.add_argument(
        "--use-tanaka",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("CHROMAGRT_USE_TANAKA", defaults["use_tanaka"]),
    )
    model.add_argument(
        "--exclude-condition-blocks",
        dest="excluded_condition_blocks",
        type=_csv_values,
        default=_env_csv("CHROMAGRT_EXCLUDE_CONDITION_BLOCKS", defaults["excluded_condition_blocks"]),
        metavar="BLOCK[,BLOCK...]",
        help="Chromatographic blocks to omit: composition, static or gradient.",
    )
    model.add_argument(
        "--exclude-molecular-descriptors",
        dest="excluded_molecular_descriptors",
        type=_csv_values,
        default=_env_csv(
            "CHROMAGRT_EXCLUDE_MOLECULAR_DESCRIPTORS",
            defaults["excluded_molecular_descriptors"],
        ),
        metavar="NAME[,NAME...]",
        help="Molecular descriptors to omit: mono_iso_mass and/or xlogp.",
    )
    model.add_argument("--dropout", type=float, default=_env_float("CHROMAGRT_DROPOUT", defaults["dropout"]))

    training = parser.add_argument_group("optimization")
    training.add_argument(
        "--lr",
        type=float,
        default=_env_float("CHROMAGRT_LR", defaults["lr"]),
        help="Initial learning rate for ReduceLROnPlateau.",
    )
    training.add_argument(
        "--plateau-factor",
        type=float,
        default=_env_float("CHROMAGRT_PLATEAU_FACTOR", defaults["plateau_factor"]),
    )
    training.add_argument(
        "--plateau-patience",
        type=int,
        default=_env_int("CHROMAGRT_PLATEAU_PATIENCE", defaults["plateau_patience"]),
    )
    training.add_argument(
        "--plateau-min-lr",
        type=float,
        default=_env_float("CHROMAGRT_PLATEAU_MIN_LR", defaults["plateau_min_lr"]),
    )
    training.add_argument("--max-epochs", type=int, default=_env_int("CHROMAGRT_MAX_EPOCHS", defaults["max_epochs"]))
    training.add_argument("--seed", type=int, default=_env_int("CHROMAGRT_SEED", defaults["seed"]))
    training.add_argument("--accelerator", default=os.environ.get("CHROMAGRT_ACCELERATOR", defaults["accelerator"]))
    return parser


def model_config_from_args(args: argparse.Namespace) -> dict:
    config = PAPER_CHROMAGRT_DEFAULTS.copy()
    for key in config:
        if hasattr(args, key):
            config[key] = getattr(args, key)
    return config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    supported_blocks = {"composition", "static", "gradient"}
    unknown_blocks = set(args.excluded_condition_blocks) - supported_blocks
    if unknown_blocks:
        parser.error(f"unsupported condition blocks: {', '.join(sorted(unknown_blocks))}")
    supported_descriptors = {"mono_iso_mass", "xlogp"}
    unknown = set(args.excluded_molecular_descriptors) - supported_descriptors
    if unknown:
        parser.error(f"unsupported molecular descriptors: {', '.join(sorted(unknown))}")
    if not 0.0 < args.plateau_factor < 1.0:
        parser.error("--plateau-factor must be between 0 and 1")
    if args.plateau_patience < 0:
        parser.error("--plateau-patience must be non-negative")
    if args.plateau_min_lr < 0.0:
        parser.error("--plateau-min-lr must be non-negative")
    return args
