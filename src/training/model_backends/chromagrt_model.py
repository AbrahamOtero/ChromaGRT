"""ChromaGRT model implemented as a Lightning module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import pytorch as pl

from src.training.model_backends.chromagrt_defaults import (
    FFN_DIM,
    HIDDEN_DIM,
    MAX_DISTANCE,
    NUM_HEADS,
    NUM_LAYERS,
    PLATEAU_COOLDOWN,
    PLATEAU_THRESHOLD,
    PLATEAU_THRESHOLD_MODE,
    WEIGHT_DECAY,
)
from src.training.model_backends.chromagrt_dataset import TANAKA_COLUMNS


class ChromaGRTEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x, attn_bias, padding_mask):
        batch_size, n_nodes = x.shape[:2]
        num_heads = attn_bias.shape[-1]
        attn_mask = attn_bias.permute(0, 3, 1, 2).reshape(batch_size * num_heads, n_nodes, n_nodes)
        key_padding_mask = torch.zeros_like(padding_mask, dtype=x.dtype)
        key_padding_mask = key_padding_mask.masked_fill(padding_mask, float("-inf"))
        attn_output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class ChromaGRTRegressor(pl.LightningModule):
    def __init__(self, condition_dim, target_mean, target_std, config):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = dict(config)
        hidden_dim = HIDDEN_DIM
        num_heads = NUM_HEADS
        dropout = self.config["dropout"]
        ffn_dim = FFN_DIM
        condition_hidden_dim = HIDDEN_DIM

        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.tanaka_block_dropout = float(
            self.config.get("tanaka_block_dropout", 0.0)
        )
        if not 0.0 <= self.tanaka_block_dropout <= 1.0:
            raise ValueError("tanaka_block_dropout must be between 0 and 1")

        self.tanaka_value_indices = tuple(
            int(index) for index in self.config.get("tanaka_value_indices", ())
        )
        self.tanaka_missing_indices = tuple(
            int(index) for index in self.config.get("tanaka_missing_indices", ())
        )
        if self.tanaka_block_dropout > 0.0:
            expected_size = len(TANAKA_COLUMNS)
            if (
                len(self.tanaka_value_indices) != expected_size
                or len(self.tanaka_missing_indices) != expected_size
            ):
                raise ValueError(
                    f"Expected {expected_size} Tanaka value indices and "
                    f"{expected_size} missingness indices"
                )
            all_indices = (
                *self.tanaka_value_indices,
                *self.tanaka_missing_indices,
            )
            if len(set(all_indices)) != len(all_indices):
                raise ValueError(
                    "Tanaka value and missingness indices must be distinct"
                )
            if any(index < 0 or index >= condition_dim for index in all_indices):
                raise ValueError("Tanaka indices fall outside the condition input")

        self.atom_encoder = nn.Embedding(119, hidden_dim, padding_idx=0)
        self.degree_encoder = nn.Embedding(6, hidden_dim, padding_idx=0)
        self.formal_charge_encoder = nn.Embedding(7, hidden_dim)
        self.hybridization_encoder = nn.Embedding(6, hidden_dim, padding_idx=0)
        self.aromatic_encoder = nn.Embedding(2, hidden_dim)
        self.total_hs_encoder = nn.Embedding(5, hidden_dim, padding_idx=0)
        self.distance_bias = nn.Embedding(MAX_DISTANCE + 1, num_heads)
        self.condition_node_projection = nn.Linear(condition_hidden_dim, hidden_dim)
        self.condition_node_bias = nn.Parameter(torch.zeros(num_heads))

        self.layers = nn.ModuleList(
            [
                ChromaGRTEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
                for _ in range(NUM_LAYERS)
            ]
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, condition_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(condition_hidden_dim, condition_hidden_dim),
            nn.GELU(),
        )

        self.film = nn.Linear(condition_hidden_dim, hidden_dim * 2)

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, 1),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.config["plateau_factor"],
            patience=self.config["plateau_patience"],
            threshold=PLATEAU_THRESHOLD,
            threshold_mode=PLATEAU_THRESHOLD_MODE,
            cooldown=PLATEAU_COOLDOWN,
            min_lr=self.config["plateau_min_lr"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_mae",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
                "name": "plateau_val_mae",
            },
        }

    def _node_features(self, batch):
        return (
            self.atom_encoder(batch["atom_type"])
            + self.degree_encoder(batch["degree"])
            + self.formal_charge_encoder(batch["formal_charge"])
            + self.hybridization_encoder(batch["hybridization"])
            + self.aromatic_encoder(batch["aromatic"])
            + self.total_hs_encoder(batch["total_hs"])
        )

    def _append_attention_node(self, attn_bias, node_bias):
        batch_size = attn_bias.shape[0]
        row_bias = node_bias.view(1, 1, 1, -1).expand(batch_size, 1, attn_bias.shape[2], -1)
        column_bias = node_bias.view(1, 1, 1, -1).expand(batch_size, attn_bias.shape[1], 1, -1)
        corner_bias = node_bias.view(1, 1, 1, -1).expand(batch_size, 1, 1, -1)
        attn_bias = torch.cat([attn_bias, row_bias], dim=1)
        return torch.cat([attn_bias, torch.cat([column_bias, corner_bias], dim=1)], dim=2)

    def _apply_tanaka_block_dropout(self, conditions):
        if not self.training or self.tanaka_block_dropout == 0.0:
            return conditions

        batch_size = conditions.shape[0]
        if self.tanaka_block_dropout == 1.0:
            dropped = torch.ones(
                batch_size,
                dtype=torch.bool,
                device=conditions.device,
            )
        else:
            dropped = (
                torch.rand(batch_size, device=conditions.device)
                < self.tanaka_block_dropout
            )

        masked_conditions = conditions.clone()
        row_mask = dropped.unsqueeze(1)

        tanaka_values = masked_conditions[:, self.tanaka_value_indices]
        masked_conditions[:, self.tanaka_value_indices] = torch.where(
            row_mask,
            torch.zeros_like(tanaka_values),
            tanaka_values,
        )

        missingness = masked_conditions[:, self.tanaka_missing_indices]
        masked_conditions[:, self.tanaka_missing_indices] = torch.where(
            row_mask,
            torch.ones_like(missingness),
            missingness,
        )
        return masked_conditions

    def _encode_conditions(self, batch):
        conditions = self._apply_tanaka_block_dropout(batch["conditions"])
        return self.condition_encoder(conditions)

    def forward(self, batch):
        x = self._node_features(batch)
        attn_bias = self.distance_bias(batch["distance"])
        padding_mask = batch["padding_mask"]
        atom_padding_mask = padding_mask
        condition_embedding = self._encode_conditions(batch)

        batch_size = x.shape[0]
        condition_node = self.condition_node_projection(condition_embedding).unsqueeze(1)
        x = torch.cat([x, condition_node], dim=1)
        condition_padding = torch.zeros((batch_size, 1), dtype=torch.bool, device=padding_mask.device)
        padding_mask = torch.cat([padding_mask, condition_padding], dim=1)
        attn_bias = self._append_attention_node(attn_bias, self.condition_node_bias)

        for layer in self.layers:
            x = layer(x, attn_bias, padding_mask)

        atom_states = x[:, :-1]
        valid_nodes = (~atom_padding_mask).unsqueeze(-1)
        graph_embedding = (atom_states * valid_nodes).sum(dim=1) / valid_nodes.sum(dim=1).clamp(min=1)
        gamma, beta = self.film(condition_embedding).chunk(2, dim=1)
        model_input = graph_embedding * (1 + gamma) + beta
        return self.regression_head(model_input)

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = F.l1_loss(pred, batch["target"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        loss = F.l1_loss(pred, batch["target"])
        mae_seconds = F.l1_loss(pred, batch["target"]) * self.target_std
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_mae", mae_seconds, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred = self(batch)
        return pred * self.target_std + self.target_mean
