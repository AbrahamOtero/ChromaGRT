"""Molecular graph featurization for ChromaGRT."""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops


HYBRIDIZATION_TO_ID = {
    Chem.rdchem.HybridizationType.SP: 1,
    Chem.rdchem.HybridizationType.SP2: 2,
    Chem.rdchem.HybridizationType.SP3: 3,
    Chem.rdchem.HybridizationType.SP3D: 4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
}


def mol_from_row(row):
    smiles = row.get("smiles.std", None)
    inchi = row.get("inchi.std", None)
    mol = None
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None and isinstance(inchi, str):
        mol = Chem.MolFromInchi(inchi, sanitize=True)
    if mol is None:
        raise ValueError("Could not build RDKit molecule from row.")
    return mol


def featurize_mol(mol, max_distance):
    atom_type = []
    degree = []
    formal_charge = []
    hybridization = []
    aromatic = []
    total_hs = []

    for atom in mol.GetAtoms():
        atom_type.append(min(atom.GetAtomicNum(), 118))
        degree.append(min(atom.GetTotalDegree(), 5))
        formal_charge.append(int(np.clip(atom.GetFormalCharge(), -3, 3)) + 3)
        hybridization.append(HYBRIDIZATION_TO_ID.get(atom.GetHybridization(), 0))
        aromatic.append(int(atom.GetIsAromatic()))
        total_hs.append(min(atom.GetTotalNumHs(), 4))

    distance = rdmolops.GetDistanceMatrix(mol).astype(np.int64)
    distance = np.clip(distance, 0, max_distance)

    return {
        "atom_type": torch.tensor(atom_type, dtype=torch.long),
        "degree": torch.tensor(degree, dtype=torch.long),
        "formal_charge": torch.tensor(formal_charge, dtype=torch.long),
        "hybridization": torch.tensor(hybridization, dtype=torch.long),
        "aromatic": torch.tensor(aromatic, dtype=torch.long),
        "total_hs": torch.tensor(total_hs, dtype=torch.long),
        "distance": torch.tensor(distance, dtype=torch.long),
    }
