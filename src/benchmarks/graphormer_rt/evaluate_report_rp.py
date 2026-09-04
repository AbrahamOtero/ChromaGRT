"""Evaluate GraphormerRT RP model and write metrics"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import site
import subprocess
import sys
import tempfile
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _cuda_library_path() -> str:
    paths: list[str] = []
    for root in site.getsitepackages():
        paths.extend(str(path) for path in Path(root).glob("nvidia/*/lib") if path.is_dir())
    return ":".join(paths)


def _link_or_copy_checkpoint(checkpoint: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(checkpoint)
    except OSError:
        shutil.copy2(checkpoint, target)


def _run_graphormer_rt_evaluator(args: argparse.Namespace, predictions_path: Path) -> None:
    bench_dir = Path(__file__).resolve().parent
    upstream_dir = bench_dir / "upstream"
    property_prediction_dir = upstream_dir / "examples" / "property_prediction"
    evaluate_py = upstream_dir / "graphormer" / "evaluate" / "evaluate.py"

    data_dir = args.data_dir.resolve()
    env = os.environ.copy()
    env["GRAPHORMER_RT_RP_CSV"] = str(args.csv or data_dir / "report_rp.csv")
    env["GRAPHORMER_RT_RP_METADATA"] = str(args.metadata or data_dir / "RP_metadata.pickle")
    env["GRAPHORMER_RT_TRAIN_FRACTION"] = str(args.train_fraction)
    env["GRAPHORMER_RT_VAL_FRACTION"] = str(args.val_fraction)
    env["GRAPHORMER_RT_SPLIT_MODE"] = args.split_mode
    env["GRAPHORMER_RT_SPLIT_SEED"] = str(args.split_seed)
    if args.manifest:
        env["GRAPHORMER_RT_MANIFEST"] = str(args.manifest.resolve())

    cuda_libs = _cuda_library_path()
    if cuda_libs:
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = cuda_libs if not existing else f"{cuda_libs}:{existing}"

    with tempfile.TemporaryDirectory(prefix="graphormer_rt_eval_") as tmp:
        checkpoint_dir = Path(tmp) / "checkpoint_dir"
        _link_or_copy_checkpoint(args.checkpoint.resolve(), checkpoint_dir / args.checkpoint.name)

        cmd = [
            sys.executable,
            str(evaluate_py),
            "--user-dir",
            str(upstream_dir / "graphormer"),
            "--num-workers",
            str(args.num_workers),
            "--ddp-backend=legacy_ddp",
            "--seed",
            str(args.seed),
            "--user-data-dir",
            "report_rp_training_dataset",
            "--dataset-name",
            "RT_Library",
            "--task",
            "graph_prediction_with_flag",
            "--criterion",
            args.criterion,
            "--arch",
            "graphormer_base",
            "--encoder-layers",
            str(args.encoder_layers),
            "--encoder-embed-dim",
            str(args.encoder_embed_dim),
            "--encoder-ffn-embed-dim",
            str(args.encoder_ffn_embed_dim),
            "--encoder-attention-heads",
            str(args.encoder_attention_heads),
            "--freeze-level",
            "0",
            "--mlp-layers",
            str(args.mlp_layers),
            "--batch-size",
            str(args.batch_size),
            "--num-classes",
            "1",
            "--save-path",
            str(predictions_path),
            "--save-dir",
            str(checkpoint_dir),
            "--split",
            args.split,
        ]

        subprocess.run(cmd, cwd=property_prediction_dir, env=env, check=True)


def _compute_metrics(predictions_path: Path) -> dict[str, float | int]:
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    relative_errors: list[float] = []

    with predictions_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            true_rt = float(row["True RT"])
            predicted_rt = float(row["Predicted RT"])
            error = true_rt - predicted_rt
            abs_error = abs(error)
            absolute_errors.append(abs_error)
            squared_errors.append(error * error)
            if abs(true_rt) > 1e-8:
                relative_errors.append(abs_error / abs(true_rt) * 100.0)

    if not absolute_errors:
        raise ValueError(f"No predictions found in {predictions_path}")

    return {
        "MAE": sum(absolute_errors) / len(absolute_errors),
        "RMSE": math.sqrt(sum(squared_errors) / len(squared_errors)),
        "MRE": sum(relative_errors) / len(relative_errors) if relative_errors else float("nan"),
        "n_samples": len(absolute_errors),
    }


def _write_metrics(output_dir: Path, metrics: dict[str, float | int], metadata: dict[str, str]) -> None:
    output = {**metadata, **metrics}

    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with (output_dir / "metrics.txt").open("w") as handle:
        handle.write(f"MAE: {metrics['MAE']:.4f} s\n")
        handle.write(f"RMSE: {metrics['RMSE']:.4f} s\n")
        handle.write(f"MRE: {metrics['MRE']:.4f} %\n")
        handle.write(f"n_samples: {metrics['n_samples']}\n")
        for key, value in metadata.items():
            handle.write(f"{key}: {value}\n")


def parse_args() -> argparse.Namespace:
    project_root = _project_root()
    default_data_dir = project_root / "data" / "benchmarks" / "graphormer_rt" / "report_rp_paper"
    default_manifest = os.environ.get("GRAPHORMER_RT_MANIFEST")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", default=default_data_dir, type=Path)
    parser.add_argument("--csv", default=None, type=Path)
    parser.add_argument("--metadata", default=None, type=Path)
    parser.add_argument("--train-fraction", default=0.8, type=float)
    parser.add_argument("--val-fraction", default=0.1, type=float)
    parser.add_argument("--split-mode", default=os.environ.get("GRAPHORMER_RT_SPLIT_MODE", "random"))
    parser.add_argument("--split-seed", default=int(os.environ.get("GRAPHORMER_RT_SPLIT_SEED", "23")), type=int)
    parser.add_argument("--manifest", default=Path(default_manifest) if default_manifest else None, type=Path)
    parser.add_argument("--num-workers", default=16, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--seed", default=23, type=int)
    parser.add_argument("--criterion", default="mse_rt")
    parser.add_argument("--encoder-layers", default=int(os.environ.get("GRAPHORMER_RT_ENCODER_LAYERS", "8")), type=int)
    parser.add_argument("--encoder-embed-dim", default=int(os.environ.get("GRAPHORMER_RT_ENCODER_EMBED_DIM", "512")), type=int)
    parser.add_argument(
        "--encoder-ffn-embed-dim",
        default=int(os.environ.get("GRAPHORMER_RT_ENCODER_FFN_EMBED_DIM", "512")),
        type=int,
    )
    parser.add_argument(
        "--encoder-attention-heads",
        default=int(os.environ.get("GRAPHORMER_RT_ENCODER_ATTENTION_HEADS", "64")),
        type=int,
    )
    parser.add_argument("--mlp-layers", default=int(os.environ.get("GRAPHORMER_RT_MLP_LAYERS", "5")), type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.csv"

    _run_graphormer_rt_evaluator(args, predictions_path)
    metrics = _compute_metrics(predictions_path)
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "split_mode": args.split_mode,
        "split_seed": str(args.split_seed),
        "manifest": str(args.manifest.resolve()) if args.manifest else "",
        "predictions": str(predictions_path),
        "encoder_layers": str(args.encoder_layers),
        "encoder_embed_dim": str(args.encoder_embed_dim),
        "encoder_ffn_embed_dim": str(args.encoder_ffn_embed_dim),
        "encoder_attention_heads": str(args.encoder_attention_heads),
        "mlp_layers": str(args.mlp_layers),
    }
    _write_metrics(output_dir, metrics, metadata)

    print(f"MAE: {metrics['MAE']:.4f} s")
    print(f"RMSE: {metrics['RMSE']:.4f} s")
    print(f"MRE: {metrics['MRE']:.4f} %")
    print(f"n_samples: {metrics['n_samples']}")
    print(f"metrics: {output_dir / 'metrics.txt'}")


if __name__ == "__main__":
    main()
