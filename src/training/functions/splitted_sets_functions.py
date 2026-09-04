"""Data preprocessing and evaluation helpers for ChromaGRT."""

#IMPORT MODULES
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

#DEFINE FUNCTIONS
def add_moldescs (df, moldesc_path):
    """
        This function adds molecular descriptors to the processed and split datasets.
        All the molecules whose monoisotopic mass and xlogp are missing will be dropped.
        Also, this columns will be added next to "formula" column, this is for to commit less changes
        into next functions
    """
    moldesc_df = pd.read_csv(moldesc_path, sep = '\t')
    temp_df = df.copy ()
    temp_df = pd.merge (temp_df, moldesc_df, left_on = "inchi.std", right_on="inchi", how="inner")
    temp_df = temp_df.drop(columns=["inchi"])
    final_df = temp_df.dropna (subset=["mono_iso_mass", "xlogp"])

    # Cut and Paste
    position = final_df.columns.get_loc("formula")
    temp_monoisomass_serie = final_df.pop ("mono_iso_mass")
    temp_xlogp_serie = final_df.pop ("xlogp")
    final_df.insert(position + 1, "mono_iso_mass", temp_monoisomass_serie)
    final_df.insert(position + 2, "xlogp", temp_xlogp_serie)
    return final_df

def get_scaled_moldescs_train (train_df):
    """
        This function scales the input train dataframe's molecular descriptors columns.
        For now only 2 are being used: isotopic molecular mass and xlogp.
    """
    moldesc_scaler = StandardScaler()
    input_data = train_df.loc [:, "mono_iso_mass":"xlogp"]
    moldesc_scaler.fit (input_data)
    columns = ["mono_iso_mass", "xlogp"]
    final_df = train_df.copy ()
    final_df [columns] = moldesc_scaler.transform(final_df[columns])
    return final_df, moldesc_scaler

def get_scaled_moldesc_testval (df, train_moldesc_scaler):
    """
        Uses the moldesc_scaler gotten from train dataset to scale molecular descriptors
        mainly for val and test datasets.
    """
    final_df = df.copy()
    columns = ["mono_iso_mass", "xlogp"]
    final_df [columns] = train_moldesc_scaler.transform(final_df[columns])
    return final_df

def get_scaled_input_train_data (train_df):
    """
    Input: The train set used for training a model.
    Output: the scaled train set and the Scaler
    """
    temp_list = []
    train_input_scaler = StandardScaler ()
    index_array = np.unique (train_df["cc_id"])
    for index in index_array: #This loop is used to get the condition from each cc (chromatography condition)
        temp_df = train_df[train_df["cc_id"] == index]
        row = temp_df.iloc[:1]
        temp_list.append (row)
    temp_df = pd.concat(temp_list)
    input_data = temp_df.loc[:, "column.length":]
    train_input_scaler.fit (input_data) #This would be returned as output.
    position = train_df.columns.get_loc("column.length")
    columns_used = list(train_df.columns [position:])
    final_columns_used =  columns_used
    final_df = train_df.copy()
    final_df [final_columns_used] = train_input_scaler.transform(final_df [final_columns_used])
    return final_df, train_input_scaler

def get_scaled_datasets (df, train_input_scaler):
    """
    Used for scaling the input (metadata and gradient data) df using the train Scaler.
    In this context, this is used to get test and val data using train_input_scaler.
    """
    position = df.columns.get_loc("column.length")
    columns_used = list(df.columns[position:])
    final_columns_used =  columns_used

    #FIXED THE SCALING PROBLEM
    final_df = df.copy()
    final_df [final_columns_used] = train_input_scaler.transform(final_df[final_columns_used])
    return final_df


def get_deterministic_input_data(df):
    """
    Apply deterministic chromatographic-condition normalization.

    Categorical and presence variables are binary, bounded quantities use
    predefined divisors, and gradient times are expressed in minutes. The
    transformation is identical for training, validation, and test data and
    does not fit split-specific statistics.
    """
    final_df = df.copy()

    def numeric(column):
        return pd.to_numeric(final_df[column], errors="coerce").fillna(0.0)

    for column in final_df.columns:
        if column == "column.length":
            final_df[column] = numeric(column) / 250.0
        elif column == "column.temperature":
            final_df[column] = numeric(column) / 100.0
        elif column.startswith("t [min]_"):
            final_df[column] = numeric(column) / 60.0
        elif column.startswith("B [%]_"):
            final_df[column] = numeric(column) / 100.0
        elif column.startswith("eluent.") and column.endswith(".pH"):
            final_df[column] = numeric(column) / 14.0
        elif column.startswith("eluent."):
            final_df[column] = (numeric(column) != 0.0).astype(float)

    return final_df


def get_res_table (test_df,
                   pred_array,
                   save_dir,
                   save_results=True,
                   using_moldescs=False,
                   ):
    """
        Update: A new Boolean save_results has been added as a parameter. This is mainly added for the update of model_per_repo/main.py to get the result table.
        As it has been set to True when not defined, other scripts that used this function do not have to be modified.
    """
    if using_moldescs:
        temp_df =  test_df [["cc_id","molecule_id", "name","mono_iso_mass", "xlogp", "inchi.std", "rt", "max_rt", "mean_rt"]]
    else:
        temp_df = test_df [["cc_id","molecule_id", "name", "inchi.std", "rt", "max_rt", "mean_rt"]]

    temp_df ["pred_rt"] = pred_array
    temp_df ["diff"] = np.abs (temp_df["pred_rt"] - temp_df["rt"])
    temp_df ["rel_error_max"] = temp_df ["diff"]*100 / temp_df["max_rt"]
    temp_df ["rel_error_mean"] = temp_df ["diff"]*100 / temp_df["mean_rt"]
    temp_df ["MRE"] = temp_df["diff"]*100 / temp_df ["rt"]
    filename = save_dir + "Results.tsv"
    if save_results:
        temp_df.to_csv(filename, sep="\t", index=False)
    return temp_df

def metrics_from_dataframe (df):
    """
    Input: DataFrame with "diff" column..
    Output: MAE and RMSE calculated from those values.
    """
    mae = np.mean (df["diff"])
    rmse = np.sqrt(np.mean (df["diff"] ** 2))
    mean_rel_error_max_rt = np.mean (df["rel_error_max"])
    mean_rel_error_mean_rt = np.mean (df["rel_error_mean"])
    mre = np.mean (df["MRE"])
    return mae, rmse, mre, mean_rel_error_max_rt, mean_rel_error_mean_rt

def write_metrics_per_cc (res_df, result_path):
    """
    Input: The Result dataframe. (Created from "get_res_table")
    Usage: Get a .tsv file containing the mean per repository data metrics got before (MAE, RMSE and %error).
    """
    result = {
        "cc":[],
        "MAE":[],
        "RMSE": [],
        "MRE":[],
        "Mean_relative_error_max":[],
        "Mean_relative_error_mean":[],
        "n molecules test": []
    }
    index_array = np.unique (res_df ["cc_id"])
    for index in index_array:
        temp_df = res_df [res_df ["cc_id"] == index]
        result ["cc"].append (index)
        result ["MAE"].append (np.mean (temp_df["diff"]))
        result ["RMSE"].append (np.sqrt (np.mean(temp_df["diff"] ** 2)))
        result ["MRE"].append (np.mean (temp_df["MRE"]))
        result ["Mean_relative_error_max"].append (np.mean (temp_df["rel_error_max"]))
        result ["Mean_relative_error_mean"].append (np.mean (temp_df["rel_error_mean"]))
        result ["n molecules test"].append (len(temp_df))
    result = pd.DataFrame(result)
    result.to_csv (result_path + "metrics_per_cc.tsv", sep="\t", index=False)
    return

def write_metric_txt (mae, rmse, mre, mean_rel_error_max_rt, mean_rel_error_mean_rt,results_path):
    filename = results_path + "metrics.txt"
    with open (filename, "w") as f:
        f.write (f'MAE: {mae:.4f} s\n'
                 f'RMSE: {rmse:.4f} s.\n'
                 f'MRE: {mre:.4f} %\n'
                 f'Relative error to max rt (%): {mean_rel_error_max_rt:.4f}\n'
                 f'Relative error to mean rt (%): {mean_rel_error_mean_rt:.4f}\n')

def write_parameters_file(param_dict, results_path):
    filename = results_path + "parameters.txt"
    with open(filename, "w") as f:
        f.write(f'Parameters used for this model:\n')
        for key, value in param_dict.items():
            f.write(f"{key}: {value}\n")
