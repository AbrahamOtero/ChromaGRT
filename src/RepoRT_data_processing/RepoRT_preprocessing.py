"""Preprocess raw RepoRT RP tables for the article data pipeline.

The module processes retention-time records, column metadata and gradients. It
optionally filters the SMRT records by Natural Product Likeness Score (NPLS). It
converts retention times and gradient times from minutes to seconds, updates
molecular formulas from SMILES or InChI, standardises column-metadata units, 
fills missing eluent compositions with zero, one-hot encodes usp.code, and replaces
missing or negative gradient flow rates with the corresponding column value.

It writes the three preprocessed tables, and their merged table to
the output directory. When NPLS filtering is enabled, it also records
the number of removed SMRT molecules.
"""


# Import modules

import os
import sys
import json

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import OneHotEncoder

from rdkit.Chem import MolFromInchi, MolFromSmiles
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import RDConfig

contrib_path = os.path.join(RDConfig.RDContribDir, 'NP_Score')

sys.path.append(contrib_path)

import npscorer

#PARAMETERS
SMRT_DIR_ID = "0186"
NPLS_THRESHOLD = np.float64(-0.6)
IMPUTE_MISSING_COLUMN_METADATA = False


# HELPER FUNCTIONS

def _get_input_df (rt_input,
                   cc_input,
                   grad_input):
    """
        This function checks for the input files and loads then as pd.DataFrames
    """
    missing = [
        str(path)
        for path in (rt_input, cc_input, grad_input)
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing raw RepoRT inputs: " + ", ".join(missing))

    print ("Fetching the input tables...")
    rt_df = pd.read_csv(rt_input, sep='\t',dtype={"dir_id":str})
    cc_df = pd.read_csv(cc_input, sep='\t', dtype={"dir_id":str})
    grad_df = pd.read_csv(grad_input, sep='\t', dtype={"dir_id":str})

    return rt_df, cc_df, grad_df

def _get_molecule_name (column_name):
    """
        Given a column name in this format: Eluent.A.mol_name
        A string only containing the mol_name will be returned.
    """
    molecule = column_name.split(".")[2]
    return str(molecule)

def _infer_t0_val (diameter, length, fr):
    """
        Used for inferring the t0. Here t0 is calculated as V0/T.
        In RepoRT, inner diameter is given in cm, length in mm and fr in mL/min.
        So we have to pass length (mm) to cm.
    """
    base_area = np.pi * (diameter / 2)**2
    return round(((0.66*base_area*length/10)/fr)/100, 5)

# RT data preprocessing functions
def _get_npls_scored_df (df) -> pd.DataFrame:
    """
        Summirizes the process for getting the NPLS for a df.
    """
    fscore = npscorer.readNPModel()
    smiles_array = df.loc[:,"smiles.std"].values
    mol_array = [ MolFromSmiles(smiles) for smiles in smiles_array]
    score_array = [ npscorer.scoreMol (mol_obj, fscore) if mol_obj is not None else np.nan for mol_obj in mol_array]
    scored_df = df.copy()
    scored_df ["NPLS"] = score_array
    return scored_df.dropna(subset=["NPLS"])

def _filter_smrt_by_npls (df, smrt_id=SMRT_DIR_ID, threshold=NPLS_THRESHOLD):
    """
        This function filters the smrt dataset by a NPLS threshold given.
    """
    temp_df = df.copy()

    if smrt_id not in (df["dir_id"].unique()):
        return temp_df

    smrt_df = temp_df[temp_df["dir_id"] == smrt_id]
    scored_smrt_df = _get_npls_scored_df(smrt_df)
    filtered_df = scored_smrt_df[scored_smrt_df["NPLS"] >= threshold]
    report_df = pd.DataFrame({"dir_id":["0186"],
                             "n molecules": [len(smrt_df) - len(filtered_df) ]})
    no_smrt_df = temp_df[temp_df["dir_id"] != smrt_id]
    final_df = pd.concat ([no_smrt_df, filtered_df], ignore_index=True)
    final_df.drop(columns = ["NPLS"], inplace = True)
    return final_df.sort_values(by="dir_id",ignore_index=True), report_df
def _transform_min2s(df):
    """
        This functions converts minutes to seconds
    """
    final_df = df.copy()
    final_df["rt"] = round(final_df["rt"] * 60, 2)

    return final_df


def _get_new_formula (df):
    """
        Input: RT dataframe with a column named "formula".
        Returns the dataframe with the "formula" column updated with formulas calculated with RDkit.
        Updated, removed both functions defined before and now list comprehension is applied.
    """
    inchi_array = df.loc[:,"inchi.std"].values
    smiles_array = df.loc[:,"smiles.std"].values
    mol_array = [ MolFromSmiles(smiles) or MolFromInchi(inchi) for smiles, inchi in zip(smiles_array, inchi_array)]
    formula_array = [ CalcMolFormula(mol_obj) if mol_obj is not None else df.loc[index, "formula"]
                      for index, mol_obj in enumerate(mol_array)]
    df ["formula"] = formula_array
    return df

def _obtain_preprocessed_rt_df (rt_df,
                                path2dir,
                                method,
                                filename="preprocessed_rt_data.tsv",
                                filter_smrt_by_npls=True):
    """
        Preprocesses the RT data and saves it as a tsv file
    """
    temp_df = rt_df.copy()
    if not filter_smrt_by_npls:
        temp_df_smrt_filtered = temp_df
    elif method == "_" or method == "RP":
        temp_df_smrt_filtered, report_df = _filter_smrt_by_npls(temp_df)
        report_df.to_csv (os.path.join(path2dir, "molecules_dropped_SMRT_byNPLS.tsv"), sep='\t', index=False)
    else:
        temp_df_smrt_filtered = _filter_smrt_by_npls(temp_df)
    print ("Converting the min to seconds...")
    temp_df_smrt_filtered = _transform_min2s(temp_df_smrt_filtered)

    print("Updating the formulas...")
    final_df = _get_new_formula(temp_df_smrt_filtered)
    final_df["dir_id"] = [str(idmol).split("_")[0] for idmol in final_df["molecule_id"]]

    path2file = os.path.join(path2dir, filename)
    print(f"Saving the preprocessed RT in {path2file}")
    final_df.to_csv(path2file, sep='\t', index=False)

    return final_df


def preprocess_rp_dataset(raw_root,
                          output_dir,
                          filter_smrt_by_npls=True):
    """Preprocess one RP dataset into an isolated output directory.

    This entry point is intended for reproducible dataset variants. It writes
    only to the supplied output directory and refuses to reuse a populated
    destination. In particular, ``filter_smrt_by_npls=False`` preserves all
    SMRT rows instead of applying the historical NPLS threshold.
    """
    raw_root = Path(raw_root)
    output_dir = Path(output_dir)
    raw_data_dir = raw_root / "raw_data"
    input_paths = {
        "rt": raw_data_dir / "raw_rt_data.tsv",
        "cc": raw_data_dir / "raw_cc_data.tsv",
        "grad": raw_data_dir / "raw_grad_data.tsv",
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RP raw inputs: {missing}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite populated preprocessing output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    rt_df, cc_df, grad_df = _get_input_df(
        rt_input=input_paths["rt"],
        cc_input=input_paths["cc"],
        grad_input=input_paths["grad"],
    )
    preprocessed_rt_df = _obtain_preprocessed_rt_df(
        rt_df=rt_df,
        path2dir=output_dir,
        method="RP",
        filter_smrt_by_npls=filter_smrt_by_npls,
    )
    preprocessed_cc_df = _obtain_preprocessed_cc_data(
        cc_df=cc_df,
        path2dir=output_dir,
    )
    preprocessed_grad_df = _preprocess_grad_data(
        grad_df=grad_df,
        imputed_cc_df=preprocessed_cc_df,
        path2dir=output_dir,
    )

    print("Making the complete preprocessed datatable...")
    rt_cc_df = pd.merge(preprocessed_rt_df, preprocessed_cc_df, on="dir_id", how="inner")
    complete_df = pd.merge(rt_cc_df, preprocessed_grad_df, on="dir_id", how="inner")
    complete_df["dir_id"] = [str(idmol).split("_")[0] for idmol in complete_df["molecule_id"]]
    complete_path = output_dir / "complete_preprocessed_data.tsv"
    print(f"Saving the complete preprocessed data as {complete_path}...")
    complete_df.to_csv(complete_path, sep="\t", index=False)

    with (output_dir / "preprocessing_config.json").open("w") as handle:
        json.dump(
            {
                "raw_root": str(raw_root.resolve()),
                "filter_smrt_by_npls": bool(filter_smrt_by_npls),
                "npls_threshold": float(NPLS_THRESHOLD) if filter_smrt_by_npls else None,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return complete_df

# CC DATA PREPROCESSING

def _process_column_data (df):
    """
    Processes a raw RepoRT metadata tsv file.
    The processing consists in:
        1. Fill all NA values with the global mean, but the column.t0 value.
        2. With all the NA vals of the metadata filled, the t0 for those columns will be inferred:
                            t0 = V0 / F = 0.66*Vcolumn / Flow_rate
    """
    if not IMPUTE_MISSING_COLUMN_METADATA:
        return df

    #Get a smaller df for faster iteration. The id column is not used.
    temp_df = df.loc [:, "column.name":"column.flowrate"]
    # Create a dictionary with the column names as keys and the GLOBAL MEANS as the values.
    means_dict = {column : round(temp_df[column].mean(), 2) for column in temp_df.columns [2:]}

    #The updating process
    for index,row in temp_df.iterrows():
        for column in temp_df.columns [2:]:
            if pd.isnull(row[column]) :
                # If the NAME AND THE COLUMN value BOTH MISSING.
                temp_df.loc[index,column] = means_dict[column] #Global mean used
            else:
                continue
    # Update the df
    df.update (temp_df)
    #Updating t0 value

    for index, row in df.iterrows():
        if row["column.t0"]==0:
            temp_t0 = _infer_t0_val(np.float64(row["column.id"]),
                                   np.float64(row["column.length"]),
                                   np.float64(row["column.flowrate"]))
            df.loc[index, "column.t0"] = temp_t0
        else:
            continue
    return df #This df contains the updated column metadata

def _process_eluent_unit (df):
    """
    Input: Requires a metadata df as input (from RepoRT). This should have all the metadata from every dataset concatenated in a single df.ç
    Output: All the unit in mM or uM converted to %(m/v) and the columns containig the ".unit" information will be dropped.
    """
    # This dictionary contains the approx. molecular weight of the molecules whose unit was expressed in "mM" or "uM"
    mws = {
        "acetic": 60,
        "phosphor": 98,
        "nh4ac": 77,
        "nh4form": 63,
        "nh4carb": 96,
        "nh4bicarb": 79,
        "nh4f": 37,
        "nh4oh": 35,
        "trieth": 101,
        "triprop": 143,
        "tribut": 185,
        "nndimethylhex": 129,
        "medronic": 176,
    }
    # The iteration is over the rows.
    for index, row in df.iterrows():
        col_index = 0
        for column in df.columns:
            col_index += 1
            if row [column] == "mM": # If the unit is "mM", we convert the value to %(m/v)
                mol_column = df.columns[col_index - 2] #Get access to the molecule's column.
                mol_name = _get_molecule_name (mol_column)
                scale_factor = mws[mol_name] / 10000 # Mw/10000
                new_value = row[df.columns[col_index -2]] * scale_factor #mM*Mw/10000
                df [mol_column] =  df [mol_column].astype(np.float64) #Necessary because the dtype in the original dset is np.int64
                df.loc[index, mol_column] = np.float64(new_value)
            elif row [column] == "µM": # If the unit is "uM", we convert the value to %(m/v)
                mol_column = df.columns[col_index - 2]
                mol_name = _get_molecule_name(mol_column)
                scale_factor = mws[mol_name] / 10000000 #The only difference here.
                new_value = row[df.columns[col_index - 2]] * scale_factor #uM*Mw/10000000
                df [mol_column] =  df [mol_column].astype(np.float64)
                df.loc[index, mol_column] = new_value
            else:
                continue
    # As all concentration data is expressed in % (m/v), the unit's columns are no longer needed, so just drop them.
    # Also the columns containing any gradient information will be dropped as we will treat them in a better way.
    drop_column_array = []
    for column in df.columns:
        if ".unit" in column or "gradient." in column:
            drop_column_array.append (column)
        else:
            continue
    df = df.drop (drop_column_array, axis =1)
    del drop_column_array
    return df #This df contains all the column metadata and eluent composition data.

def _get_one_hot_encoded_df (df):
    """
    Input: The metadata of RepoRT (processed previously or not) containing the column "column.usp.code".
    Output: An updated df with new columns of USP code one-hot encoded.
    """
    encoder = OneHotEncoder()
    one_hot_data = encoder.fit_transform(df[["column.usp.code"]])
    one_hot_df = pd.DataFrame(one_hot_data.toarray(),
                              columns=encoder.get_feature_names_out(['column.usp.code']))
    position_column_name = df.columns.get_loc("column.name")
    updated_df = pd.concat([df.iloc[:, :position_column_name + 1],
                                      one_hot_df,
                                      df.iloc[:, position_column_name + 1:]], axis=1)
    del one_hot_df
    return updated_df

def _obtain_preprocessed_cc_data(cc_df,
                                 path2dir,
                                 filename="preprocessed_cc_data.tsv"):
    """
        This function preprocesses the cc data and save it the output directory.
        This fills all the NaN values found in the eluent data part
    """

    print("Preprocessing cc data...")

    cc_df = _process_column_data(cc_df)
    cc_df= _process_eluent_unit(cc_df)
    cc_df = _get_one_hot_encoded_df(cc_df)
    filled_df = cc_df.loc [:, "eluent.A.h2o":].fillna(0)
    cc_df.loc[:, "eluent.A.h2o":] = filled_df.values

    path2file = os.path.join(path2dir, filename)

    print(f"Exporting the preprocessed cc data as {path2file}...")
    cc_df.to_csv(path2file, sep='\t', index=False)

    return cc_df


# TREATING GRADIENT INFORMATION

def _need_update (time_col, fr_col):
    """
        Outputs an Boolean depending on the columns given
    """
    return pd.notna(time_col) and pd.isna(fr_col)


def _preprocess_grad_data (grad_df,
                           imputed_cc_df,
                           path2dir,
                           filename="preprocessed_gradient_data.tsv"):
    """
        Update the flow rate colum with the imputed flow rate value.
        Transforms the time from minutes to seconds.
    """
    print("Preprocessing the gradient data...")

    grad_time_array = [ column for column in grad_df.columns if "t [min]" in column]
    grad_fr_array = [ column for column in grad_df.columns if "flow rate" in column]

    grad_df [grad_fr_array] = grad_df[grad_fr_array].apply(pd.to_numeric, errors='coerce')
    for index, row in grad_df.iterrows():
        dir_id = row ["dir_id"]
        fr = imputed_cc_df[imputed_cc_df["dir_id"] == dir_id].loc[:,"column.flowrate"].values[0]
        for grad_time, grad_fr in zip(grad_time_array, grad_fr_array):
            temp_grad_time_val = row[grad_time]
            temp_grad_fr_val = row[grad_fr]
            if _need_update (temp_grad_time_val, temp_grad_fr_val) or temp_grad_fr_val < 0: #Negative values treating
                grad_df.loc[index, grad_fr] = fr
            else:
                continue
    grad_df.loc[:,grad_time_array] *= 60

    path2file = os.path.join(path2dir, filename)
    print(f"Saving the preprocessed gradient data as {path2file}...")
    grad_df.to_csv(path2file, sep='\t', index=False)

    return grad_df
