"""Dataset and batching helpers for ChromaGRT."""

from dataclasses import dataclass
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from src.training.model_backends.chromagrt_featurizer import featurize_mol, mol_from_row


GRADIENT_COLUMN_RE = re.compile(r"^(t \[min\]|B \[%\]|flow rate \[ml/min\])_(\d+)$")
TANAKA_COLUMNS = (
    "tanaka_kpb",
    "tanaka_alpha_ch2",
    "tanaka_alpha_t_o",
    "tanaka_alpha_c_p",
    "tanaka_alpha_b_p",
    "tanaka_alpha_b_p_1",
)
TANAKA_MISSING_COLUMNS = tuple(f"{column}_is_missing" for column in TANAKA_COLUMNS)
STATIC_CONDITION_COLUMNS = (
    "column.length",
    "column.id",
    "column.particle.size",
    "column.temperature",
    "column.flowrate",
    "column.t0",
    "eluent.A.pH",
    "eluent.B.pH",
)
EXCLUDABLE_CONDITION_BLOCKS = frozenset({"composition", "static", "gradient"})
MOLECULAR_DESCRIPTOR_COLUMNS = ("mono_iso_mass", "xlogp")
EXCLUDABLE_MOLECULAR_DESCRIPTORS = frozenset(MOLECULAR_DESCRIPTOR_COLUMNS)


def _normalize_excluded_condition_blocks(excluded_blocks):
    """Validate the optional chromatography blocks removed for an ablation."""
    if excluded_blocks is None:
        return frozenset()
    if isinstance(excluded_blocks, str):
        excluded_blocks = excluded_blocks.split(",")

    normalized = frozenset(
        str(block).strip().lower()
        for block in excluded_blocks
        if str(block).strip()
    )
    unknown_blocks = normalized - EXCLUDABLE_CONDITION_BLOCKS
    if unknown_blocks:
        supported = ", ".join(sorted(EXCLUDABLE_CONDITION_BLOCKS))
        requested = ", ".join(sorted(unknown_blocks))
        raise ValueError(
            f"Unsupported excluded condition blocks: {requested}. "
            f"Supported blocks are: {supported}."
        )
    return normalized


def _normalize_excluded_molecular_descriptors(excluded_descriptors):
    """Validate molecular descriptors removed for a descriptor ablation."""
    if excluded_descriptors is None:
        return frozenset()
    if isinstance(excluded_descriptors, str):
        excluded_descriptors = excluded_descriptors.split(",")

    normalized = frozenset(
        str(descriptor).strip().lower()
        for descriptor in excluded_descriptors
        if str(descriptor).strip()
    )
    unknown_descriptors = normalized - EXCLUDABLE_MOLECULAR_DESCRIPTORS
    if unknown_descriptors:
        supported = ", ".join(sorted(EXCLUDABLE_MOLECULAR_DESCRIPTORS))
        requested = ", ".join(sorted(unknown_descriptors))
        raise ValueError(
            f"Unsupported excluded molecular descriptors: {requested}. "
            f"Supported descriptors are: {supported}."
        )
    return normalized


def _condition_block(column):
    if _is_indicator_condition_column(column):
        return "composition"
    if column in STATIC_CONDITION_COLUMNS:
        return "static"
    if _gradient_column_info(column) is not None:
        return "gradient"
    return None


def get_condition_columns(
    df,
    using_moldescs=False,
    include_tanaka=True,
    excluded_blocks=(),
    excluded_molecular_descriptors=(),
):
    excluded_blocks = _normalize_excluded_condition_blocks(excluded_blocks)
    excluded_molecular_descriptors = _normalize_excluded_molecular_descriptors(
        excluded_molecular_descriptors
    )
    columns = df.loc[:, "column.usp.code_L1":].columns.tolist()
    if not include_tanaka:
        ignored_columns = set(TANAKA_COLUMNS) | set(TANAKA_MISSING_COLUMNS)
        columns = [column for column in columns if column not in ignored_columns]
    if excluded_blocks:
        columns = [
            column
            for column in columns
            if _condition_block(column) not in excluded_blocks
        ]
    if using_moldescs:
        columns = columns + [
            column
            for column in MOLECULAR_DESCRIPTOR_COLUMNS
            if column not in columns and column not in excluded_molecular_descriptors
        ]
    return columns


def _gradient_column_info(column):
    match = GRADIENT_COLUMN_RE.match(column)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _is_indicator_condition_column(column):
    if column.startswith("column.usp.code_"):
        return True
    if column.startswith("eluent.") and not column.endswith(".pH"):
        return True
    return False


@dataclass
class TargetScaler:
    mean: float
    std: float

    def transform(self, values):
        return (values - self.mean) / self.std


class ChromaGRTDataset(Dataset):
    def __init__(self, df, condition_columns, target_scaler, max_distance=8):
        self.df = df.reset_index(drop=True)
        self.condition_columns = condition_columns
        self.target_scaler = target_scaler
        self.max_distance = max_distance
        self.graph_cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_key = row.get("smiles.std", row.get("inchi.std", idx))
        if cache_key not in self.graph_cache:
            self.graph_cache[cache_key] = featurize_mol(mol_from_row(row), self.max_distance)

        graph = self.graph_cache[cache_key]
        conditions = row.loc[self.condition_columns].to_numpy(dtype=np.float32)
        rt = np.float32(self.target_scaler.transform(row["rt"]))
        item = {
            **graph,
            "conditions": torch.tensor(conditions, dtype=torch.float32),
            "target": torch.tensor([rt], dtype=torch.float32),
        }
        return item


def build_target_scaler(train_df):
    rt_values = train_df["rt"].to_numpy(dtype=np.float32)
    std = float(rt_values.std())
    if std == 0:
        std = 1.0
    return TargetScaler(mean=float(rt_values.mean()), std=std)


def collate_chromagrt(batch):
    batch_size = len(batch)
    max_nodes = max(item["atom_type"].shape[0] for item in batch)
    condition_dim = batch[0]["conditions"].shape[0]

    atom_type = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    degree = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    formal_charge = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    hybridization = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    aromatic = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    total_hs = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    distance = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.long)
    padding_mask = torch.ones((batch_size, max_nodes), dtype=torch.bool)
    conditions = torch.zeros((batch_size, condition_dim), dtype=torch.float32)
    target = torch.zeros((batch_size, 1), dtype=torch.float32)
    for i, item in enumerate(batch):
        n_atoms = item["atom_type"].shape[0]
        atom_type[i, :n_atoms] = item["atom_type"]
        degree[i, :n_atoms] = item["degree"]
        formal_charge[i, :n_atoms] = item["formal_charge"]
        hybridization[i, :n_atoms] = item["hybridization"]
        aromatic[i, :n_atoms] = item["aromatic"]
        total_hs[i, :n_atoms] = item["total_hs"]
        distance[i, :n_atoms, :n_atoms] = item["distance"]
        padding_mask[i, :n_atoms] = False
        conditions[i] = item["conditions"]
        target[i] = item["target"]

    collated = {
        "atom_type": atom_type,
        "degree": degree,
        "formal_charge": formal_charge,
        "hybridization": hybridization,
        "aromatic": aromatic,
        "total_hs": total_hs,
        "distance": distance,
        "padding_mask": padding_mask,
        "conditions": conditions,
        "target": target,
    }
    return collated
