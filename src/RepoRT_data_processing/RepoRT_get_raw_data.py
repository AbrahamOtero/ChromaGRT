"""Download and organise the raw RepoRT data.

The downloader retrieves retention-time records, column metadata, gradients and
method metadata for the requested RepoRT directories.  It preserves the values
provided by RepoRT; curation and imputation are performed by later pipeline
stages.

It writes the following files to ``path2res``:

* ``raw_rt_data.tsv``
* ``raw_cc_data.tsv``
* ``raw_grad_data.tsv``

Also writes the raw tables separately under ``RepoRT_RP`` and
``RepoRT_HILIC``.
"""

# IMPORT MODULES

import os
import urllib.error

import numpy as np
import pandas as pd

# INTERNAL VARIABLES
SEED_URL = "https://raw.githubusercontent.com/michaelwitting/RepoRT/refs/heads/master/processed_data/"
PATH2RES = os.path.join(".", "data", "RepoRT", "raw_data/")
REPOS = np.arange(1, 393)
SPLIT_METHOD_TYPES = ("RP", "HILIC")


def repo_rt_processed_data_url(revision):
    """Return the immutable GitHub raw-data URL for one RepoRT revision."""
    revision = str(revision).strip()
    if not revision:
        raise ValueError("RepoRT revision must not be empty")
    return f"https://raw.githubusercontent.com/michaelwitting/RepoRT/{revision}/processed_data/"

# HELPER FUNCTIONS

def _num2index (repos_array):
    """
        This function convert an array containing number to RepoRT index:
        For example: 1 -> 0001
    """
    index_array=[]

    for repo in repos_array:
        index = str(repo)
        while len(index) < 4:
            index = "0" + index
        index_array.append (index)
    return np.array(index_array)

# FUNCTIONS TO FETCH DATA
def _fetch_raw_rt_data (seed_url=SEED_URL,
                       repos_array=REPOS,
                       path2res=PATH2RES,
                       filename="raw_rt_data.tsv"):
    """
        Fetch retention-time records and molecular information for the requested
        RepoRT directories.
    """

    temp_df_array = []
    index_array = _num2index (repos_array)
    for index in index_array:
        can_url = f"{seed_url}{index}/{index}_rtdata_canonical_success.tsv"
        iso_url = f"{seed_url}{index}/{index}_rtdata_isomeric_success.tsv"
        print(f"Fetching RT data for nº{index}...")
        try:
            temp_dataframe_can = pd.read_csv(can_url, sep="\t", encoding="utf-8")
            # This nested try statement avoids a not frequent but has cases where isomeric file does not exist at all, but the canonical exists.
            try:
                temp_dataframe_iso = pd.read_csv(iso_url, sep="\t", encoding="utf-8")
                temp_dataframe_iso = temp_dataframe_iso.set_index("id")
                temp_dataframe_can = temp_dataframe_can.set_index("id")
                temp_dataframe_can.update(temp_dataframe_iso)
                temp_dataframe_can = temp_dataframe_can.reset_index()
            except (urllib.error.HTTPError, pd.errors.EmptyDataError):
                print(f"Isomeric data is not found for repo nº{index}")
            temp_df_array.append(temp_dataframe_can)
        except (urllib.error.HTTPError, pd.errors.EmptyDataError):
            print(f"The repo nº {index} has not been found in the dataset. It will be skipped...")

    #Get the final array with dir_id column in the first column
    final_df = pd.concat(temp_df_array, ignore_index=True)
    dir_id_array = [str(idmol).split("_")[0] for idmol in final_df["id"]]
    final_df.insert(0, "dir_id", dir_id_array)
    final_dataframe = final_df.rename(columns={"id": "molecule_id"})

    # Write the final outputs:
    rt_data_file = os.path.join(path2res,filename)
    final_dataframe.to_csv(rt_data_file, sep="\t", index=False)
    return final_dataframe

def _fetch_raw_column_metadata(seed_url=SEED_URL,
                              repos_array=REPOS,
                              path2res=PATH2RES,
                              filename="raw_cc_data.tsv"):
    """
        Fetch column metadata for the requested RepoRT directories.
    """
    cc_array = []
    index_array = _num2index (repos_array)
    existing_index_array = []

    for index in index_array:
        cc_url = f"{seed_url}{index}/{index}_metadata.tsv"
        try:
            print(f"Fetching column metadata for nº{index}...")
            cc_df = pd.read_csv(cc_url, sep="\t", encoding="utf-8")
            cc_array.append(cc_df)
            existing_index_array.append(index)
        except (urllib.error.HTTPError, pd.errors.EmptyDataError):
            print (f"The repo nº {index} has not been found in the dataset or it is empty. It will be be skipped...")

    final_df = pd.concat(cc_array, ignore_index=True)
    final_df ["id"] = existing_index_array
    final_df.rename (columns={"id": "dir_id"}, inplace=True)

    # Write the output files
    path2file = os.path.join(path2res,filename)
    final_df.to_csv(path2file, sep="\t", index=False)

    return final_df

def _fetch_raw_gradient_data(seed_url=SEED_URL,
                            repos_array=REPOS,
                            path2res=PATH2RES,
                            filename="raw_grad_data.tsv"):
    """
        Fetch and flatten the gradient table for each requested RepoRT directory.
    """
    grad_array = []
    index_array = _num2index (repos_array)
    for index in index_array:
        print(f"Fetching gradient for {index}")
        grad_url = f'{seed_url}{index}/{index}_gradient.tsv'
        try:
            gd_df = pd.read_csv(grad_url, sep="\t", encoding="utf-8")
            if len(gd_df.index) == 0:
                print (f"The gradient_data for {index} is empty.")
            elif gd_df ["t [min]"].isna().any():
                print(f"The gradient data for {index} missing the time. It will not be used.")
            else:
                final_gd_row_df = pd.DataFrame()
                print(f"The gradient data for repo {index} will be added...")
                if gd_df.shape [1] < 5:
                    gd_df["C [%]"] = np.zeros(gd_df.shape[0])
                    gd_df["D [%]"] = np.zeros(gd_df.shape[0])
                    gd_df = gd_df[["t [min]", "A [%]", "B [%]", "C [%]", "D [%]", "flow rate [ml/min]"]]
                if gd_df["flow rate [ml/min]"].isna().any():
                    print(f"The repo {index} is missing the flow rate")
                for grad_index, row in gd_df.iterrows():
                    temp_dict = {}
                    for column in gd_df.columns:
                        temp_dict [f"{column}_{grad_index}"] =row[column]
                    row_gd_df = pd.DataFrame([temp_dict])
                    final_gd_row_df = pd.concat([final_gd_row_df, row_gd_df],  axis = 1)
                final_gd_row_df.insert(0, "dir_id", index)
                grad_array.append(final_gd_row_df)
        except urllib.error.HTTPError:
            print (f"The gradient data for repo nº {index} has not been found in the dataset. And it will be skipped...")
    final_df = pd.concat(grad_array, ignore_index=True)

    # Write the output files
    grad_file = os.path.join(path2res,filename)
    final_df.to_csv(grad_file, sep="\t", index=False)

    return final_df

def _fetch_info_data(seed_url = SEED_URL,
                     repos_array=REPOS):
    """
        Fetch method types from RepoRT and return their directory identifiers.
    """
    final_dict = {}
    index_array = _num2index(repos_array)
    for index in index_array:
        info_url = f'{seed_url}{index}/{index}_info.tsv'
        print (f"Fetching info data for {index}...")
        try:
            info_df = pd.read_csv(info_url, sep="\t", encoding="utf-8")
            chromatography_type = str(info_df ["method.type"] [0])
            if chromatography_type not in final_dict:
                final_dict[chromatography_type] = [index]
            else:
                final_dict[chromatography_type].append(index)
        except (urllib.error.HTTPError, pd.errors.EmptyDataError):
            print(f"The repo nº {index} has not been found in the dataset. And it will be skipped...")

    return final_dict

# THE USEFUL FUNCTION
def merge_complete_file(
    path2res=PATH2RES,
    seed_url=SEED_URL,
    repos_array=REPOS,
    split_output_root="./data",
):
    """
    Download the raw RepoRT tables and write separate RP and HILIC subsets.
    """
    print(f"Making the output directory: {path2res}...")
    os.makedirs(path2res, exist_ok=True)

    rt_data = _fetch_raw_rt_data(
        seed_url=seed_url,
        repos_array=repos_array,
        path2res=path2res,
    )
    cc_data = _fetch_raw_column_metadata(
        seed_url=seed_url,
        repos_array=repos_array,
        path2res=path2res,
    )
    grad_data = _fetch_raw_gradient_data(
        seed_url=seed_url,
        repos_array=repos_array,
        path2res=path2res,
    )
    info_data = _fetch_info_data(
        seed_url=seed_url,
        repos_array=repos_array,
    )
    for key in SPLIT_METHOD_TYPES:
        index_array = info_data.get(key, [])
        if not index_array:
            continue
        new_path2dir = os.path.join(split_output_root, f"RepoRT_{key}", "raw_data/")
        os.makedirs(new_path2dir, exist_ok=True)
        temp_rt_df = rt_data[rt_data["dir_id"].isin(index_array)]
        temp_cc_df = cc_data[cc_data["dir_id"].isin(index_array)]
        temp_grad_df = grad_data[grad_data["dir_id"].isin(index_array)]

        temp_rt_df.to_csv(os.path.join(new_path2dir, "raw_rt_data.tsv"), sep="\t", index=False)
        temp_cc_df.to_csv(os.path.join(new_path2dir, "raw_cc_data.tsv"), sep="\t", index=False)
        temp_grad_df.to_csv(os.path.join(new_path2dir, "raw_grad_data.tsv"), sep="\t", index=False)
