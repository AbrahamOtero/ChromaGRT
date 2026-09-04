"""Builds the molecular-descriptor tables for the ChromaGRT datasets.

For every unique standard InChI in the RepoRT table, the module
retrieves monoisotopic mass and XlogP from PubChem. Missing PubChem values are
completed with RDKit. 

"""

# Import modules

import os

import pubchempy as pcp
import numpy as np
import pandas as pd

from pathlib import Path

from rdkit.Chem import MolFromInchi
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt

# Parameters

RAW_REPORT_FILE = os.path.join (".", "data", "RepoRT", "raw_data", "raw_rt_data.tsv")
PATH2FILE = os.path.join (".", "data", "moldescs.tsv")
PATH2COMPLETE_FILE = os.path.join (".", "data", "complete_moldesc.tsv")

# Define the functions
def _pubchem_moldesc_query (path2raw_report=RAW_REPORT_FILE,
                            save_file = PATH2FILE) -> None:
    """
        This function can be used as the first part to complete the initial moldesc fetching.
        Finally, this function writes a .tsv files containing 3 columns: Inchi, monoisotopic_mass, xlogp.
        This function has been defined as PubChem uses a slightly better algorithm (XLogP 3.0) than RDkit.
    """
    if not Path(path2raw_report).exists():
        raise FileNotFoundError(f"Raw RepoRT table does not exist: {path2raw_report}")
    df = pd.read_csv(path2raw_report, sep="\t")
    inchi_array = df.loc[:, "inchi.std"].values
    mono_mw_dict = {}
    xlogp_dict = {}
    for inchi in inchi_array:
        if inchi not in mono_mw_dict.keys():
            try:
                print(f"Fetching mol descs for molecule {inchi}")
                molecule = pcp.get_compounds(str(inchi), namespace="inchi")
                mono_mw = molecule[0].monoisotopic_mass
                xlogp = molecule[0].xlogp
                mono_mw_dict[inchi] = mono_mw
                xlogp_dict[inchi] = xlogp
            except:  # This except clause is for not found molecules in PubChem.
                mono_mw_dict[inchi] = np.nan
                xlogp_dict[inchi] = np.nan
        else:  # If the molecule has already been fetched.
            print(f"The molecule: {inchi}'s has already been fetched. It will be skipped.")
            continue
    #Build the final dataframe usign the information fetched. Both dictionary have the same dimension, as np.nan has been used for filling any missing values.
    final_df = pd.DataFrame({"inchi": mono_mw_dict.keys(),
                             "mono_iso_mass": mono_mw_dict.values(),
                             "xlogp": xlogp_dict.values()})
    final_df.to_csv (save_file, sep = "\t", index=False)


def complete_moldesc_query (initial_file = PATH2FILE,
                            path2res = PATH2COMPLETE_FILE,
                            raw_report_file = RAW_REPORT_FILE)-> None:
    """
        This function requires an initial file, containing inchi and moldescs fetched with PubChem.
        If not exists, it will be built.
        This completes the moldesc search by using RDkit, and writes another .tsv file.
    """

    # Check for the initial dataframe
    if not Path (initial_file).exists():
        _pubchem_moldesc_query(path2raw_report=raw_report_file, save_file=initial_file)

    df = pd.read_csv (initial_file, sep="\t")

    for index, row in df.iterrows():
        if pd.isnull (row["mono_iso_mass"]) or pd.isnull (row["xlogp"]):
            inchi = row["inchi"]
            mol = MolFromInchi(inchi) or MolFromInchi(inchi, sanitize=False)
            mol_wt = ExactMolWt (mol)
            logp = MolLogP(mol)
            df.loc[index, "mono_iso_mass"] = round(mol_wt, 9)
            df.loc[index, "xlogp"] = round(logp, 1)
        else:
            continue
    df.to_csv (path2res, sep = "\t", index=False)
