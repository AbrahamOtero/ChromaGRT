"""Training adapter for ChromaGRT."""

import os

import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from src.training.model_backends.chromagrt_dataset import (
    ChromaGRTDataset,
    build_target_scaler,
    collate_chromagrt,
    get_condition_columns,
)
from src.training.model_backends.chromagrt_defaults import (
    DEFAULT_CHROMAGRT_CONFIG,
    EARLY_STOPPING_PATIENCE,
    MAX_DISTANCE,
)
from src.training.model_backends.chromagrt_model import ChromaGRTRegressor

def _config_from_param_dict(param_dict):
    config = DEFAULT_CHROMAGRT_CONFIG.copy()
    config.update(param_dict)
    return config


def _build_loader(df, condition_columns, target_scaler, config, shuffle):
    dataset = ChromaGRTDataset(
        df,
        condition_columns=condition_columns,
        target_scaler=target_scaler,
        max_distance=MAX_DISTANCE,
    )
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        collate_fn=collate_chromagrt,
    )


def train_chromagrt(train_df, val_df, param_dict, results_path, using_moldescs=False, save_model=True):
    config = _config_from_param_dict(param_dict)
    seed = config.get("seed")
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)
    monitor_name = "val_mae"
    include_tanaka = config.get("use_tanaka", True)
    excluded_condition_blocks = config.get("excluded_condition_blocks", ())
    excluded_molecular_descriptors = config.get("excluded_molecular_descriptors", ())
    condition_columns = get_condition_columns(
        train_df,
        using_moldescs=using_moldescs,
        include_tanaka=include_tanaka,
        excluded_blocks=excluded_condition_blocks,
        excluded_molecular_descriptors=excluded_molecular_descriptors,
    )
    target_scaler = build_target_scaler(train_df)
    train_loader = _build_loader(
        train_df,
        condition_columns,
        target_scaler,
        config,
        shuffle=True,
    )
    val_loader = _build_loader(
        val_df,
        condition_columns,
        target_scaler,
        config,
        shuffle=False,
    )

    model = ChromaGRTRegressor(
        condition_dim=len(condition_columns),
        target_mean=target_scaler.mean,
        target_std=target_scaler.std,
        config=config,
    )

    es_cb = EarlyStopping(monitor=monitor_name, mode="min", patience=EARLY_STOPPING_PATIENCE)
    checkpoint_cb = ModelCheckpoint(
        dirpath=results_path,
        filename=f"best-chromagrt-{{epoch}}-{{{monitor_name}:.4f}}",
        monitor=monitor_name,
        mode="min",
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    csv_logger = CSVLogger(save_dir=results_path, name="lightning_logs")
    trainer = pl.Trainer(
        logger=csv_logger,
        enable_progress_bar=True,
        log_every_n_steps=1,
        accelerator=config["accelerator"],
        devices=1,
        max_epochs=config["max_epochs"],
        callbacks=[es_cb, checkpoint_cb, lr_monitor],
    )
    trainer.fit(model, train_loader, val_loader)
    if checkpoint_cb.best_model_path:
        model = ChromaGRTRegressor.load_from_checkpoint(
            checkpoint_cb.best_model_path,
            condition_dim=len(condition_columns),
            target_mean=target_scaler.mean,
            target_std=target_scaler.std,
            config=config,
        )
    if save_model:
        torch.save(model.state_dict(), os.path.join(results_path, "model.pt"))
    for ckpt in os.listdir(results_path):
        if ckpt.endswith(".ckpt"):
            os.unlink(os.path.join(results_path, ckpt))

    return {
        "backend_name": "chromagrt",
        "model": model,
        "trainer": trainer,
        "target_scaler": target_scaler,
        "condition_columns": condition_columns,
        "config": config,
    }


def predict_chromagrt(model_bundle, test_df):
    test_loader = _build_loader(
        test_df,
        condition_columns=model_bundle["condition_columns"],
        target_scaler=model_bundle["target_scaler"],
        config=model_bundle["config"],
        shuffle=False,
    )
    pred = model_bundle["trainer"].predict(model_bundle["model"], test_loader)
    return np.concatenate([batch.cpu().numpy() for batch in pred], axis=0)
