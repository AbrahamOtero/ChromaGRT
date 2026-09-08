# ChromaGRT

ChromaGRT is a graph-transformer model for predicting reversed-phase liquid-
chromatography retention times from molecular structure and chromatographic
conditions. The chromatographic representation contains mobile- and stationary-
phase composition, physical and operating variables, Tanaka descriptors, and
the gradient program. It is incorporated as a global graph node and through
feature-wise linear modulation (FiLM).

This repository contains the code used for the ChromaGRT experiments and the
GraphormerRT comparison. Generated datasets, trained models, experiment logs,
and the article sources are intentionally not versioned. A complete, immutable
release of the curated ChromaGRT data is available through Zenodo.

## Repository layout

- `src/RepoRT_data_processing`: RepoRT download, preprocessing, and curation.
- `src/training/model_backends`: ChromaGRT graph construction, model, and trainer.
- `src/training/RepoRT/predefined_split`: command-line entry point for one predefined
  train/validation/test split.
- `src/benchmarks/shared_folds`: construction and evaluation of the shared random,
  Bemis--Murcko, and chromatographic-condition folds.
- `fold_manifests`: exact article fold assignments, versioned independently of
  the separately distributed curated dataset.
- `src/benchmarks/graphormer_rt`: GraphormerRT adapter and local copy of the
  upstream code.

## ChromaGRT environment

Create the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate chromagrt
```

GraphormerRT uses a separate Python 3.9 environment because of its upstream
Fairseq dependencies; see the [GraphormerRT README](src/benchmarks/graphormer_rt/README.md).

## Download curated data from Zenodo (recommended)

The Zenodo release provides the complete curated datasets required to train
ChromaGRT directly. Download `chromagrt_article_curated_data_v1.zip` from the
Zenodo release and extract it at the repository root:

```bash
unzip chromagrt_article_curated_data_v1.zip
```

The archive exposes the fixed molecular-descriptor table
(`data/complete_moldesc.tsv`) and the predefined ChromaGRT splits for the
random, Bemis--Murcko, chromatographic-condition, NPLS-filtered SMRT, and
unfiltered SMRT scenarios. These split tables already contain the filtered and
preprocessed retention-time, chromatographic-condition, gradient, column, and
Tanaka information required by the model.

We recommend using the Zenodo release for training and article reproduction.
It avoids downloading RepoRT, rerunning filtering and preprocessing,
reconstructing column information, querying PubChem, and calculating RDKit
fallback descriptors.

## Optional: rebuild the RepoRT RP data pipeline locally

Skip this section if you have downloaded and extracted the Zenodo release.
Local reconstruction can be slow. Downloading RepoRT requires substantial
time, and the subsequent PubChem queries require even more
time because they are performed for individual molecular descriptors. The
query volume can also trigger PubChem API rate limiting or temporary blocking.
For these reasons, we recommend using the Zenodo release.

The local reconstruction is intended for regenerating the curated data from
source. It fixes the RepoRT revision, repository IDs, data variants, and fold
manifests used in the article. Run its single entry point from the repository
root:

```bash
conda run --no-capture-output -n chromagrt \
  python -m src.RepoRT_data_processing.prepare_reportrp
```

That command:

- Downloads RepoRT.
- Builds the no-SMRT, NPLS-filtered SMRT, and unfiltered SMRT dataset variants.
- Preprocesses retention times, column metadata, and gradient programs.
- Groups equivalent chromatographic conditions, resolves duplicate observations,
  and applies the curation filters.
- Extracts column properties, adds Tanaka descriptors, and applies the fold manifests.
- Retrieves monoisotopic mass and logP from PubChem, using RDKit as a fallback
  when necessary.
- Generates the ChromaGRT split tables and the corresponding GraphormerRT
  assets and fold manifests.

To write the reconstructed data outside the default `data/` directory, provide
an explicit root:

```bash
conda run --no-capture-output -n chromagrt \
  python -m src.RepoRT_data_processing.prepare_reportrp \
  --data-root /path/to/data
```


Use `--skip-download` only when the exact raw RepoRT tree has already been
placed under `<data-root>/RepoRT/raw_data` and
`<data-root>/RepoRT_RP/raw_data`, where `<data-root>` is the default `data/`
directory or the path supplied with `--data-root`.

## Train models from prepared data

The following commands work once the data are available, whether they came
from the Zenodo release or from a local reconstruction. Set `DATA_ROOT` to the
directory that contains `complete_moldesc.tsv` and `benchmarks/article-v1`:

```bash
DATA_ROOT=data                              # Zenodo archive extracted at the repository root
# DATA_ROOT=/path/to/reconstructed/data     # Local reconstruction with --data-root
```

### ChromaGRT

The default configuration uses dropout 0.05, Tanaka descriptors, MAE optimization with AdamW
(`lr=5e-5`, `weight_decay=0.01`), validation-MAE checkpointing, early stopping
with patience 30, seed 42, and a maximum of 250 epochs. To inspect the current
configuration without training:

```bash
python -m src.training.RepoRT.predefined_split.main --print-config
```

Train and evaluate one predefined split:

```bash
python -m src.training.RepoRT.predefined_split.main \
  --molecular-descriptors-path "$DATA_ROOT/complete_moldesc.tsv" \
  --shared-split-dir "$DATA_ROOT/benchmarks/article-v1/random/chromagrt_splits/fold_0" \
  --results-dir logs/benchmarks/chromagrt_random/fold_0
```

Run all 10 folds:

```bash
REPORT_MOLDESC_PATH="$DATA_ROOT/complete_moldesc.tsv" \
src/benchmarks/shared_folds/scripts/run_chromagrt_kfold.sh \
  "$DATA_ROOT/benchmarks/article-v1/random" \
  --condition-normalization standard \
  --dropout 0.05 \
  --use-tanaka
```

The complete workflow and the input-ablations are documented
in [the shared-fold benchmark README](src/benchmarks/shared_folds/README.md).

### GraphormerRT

The Zenodo archive provides data for ChromaGRT only. Running GraphormerRT
requires its separate Python 3.9 environment, a complete local reconstruction
of the data pipeline, and the GraphormerRT asset-preparation steps described in
the [GraphormerRT README](src/benchmarks/graphormer_rt/README.md). Once those
steps have created the fold assets under `$DATA_ROOT/benchmarks/article-v1`, run
a scenario as follows:

```bash
bash src/benchmarks/graphormer_rt/scripts/run_graphormer_rt_kfold_fold_imputed.sh \
  "$DATA_ROOT/benchmarks/article-v1/random"
```
