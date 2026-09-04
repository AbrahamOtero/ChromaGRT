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
release of the curated ChromaGRT data will be deposited in Zenodo.

## Repository layout

- `src/RepoRT_data_processing`: RepoRT acquisition, preprocessing, and curation.
- `src/training/model_backends`: ChromaGRT graph construction, model, and trainer.
- `src/training/RepoRT/predefined_split`: command-line entry point for one predefined
  train/validation/test split.
- `src/benchmarks/shared_folds`: construction and evaluation of the shared random,
  Bemis--Murcko, and chromatographic-condition folds.
- `fold_manifests`: exact article fold assignments, versioned independently of
  the separately distributed curated dataset.
- `src/benchmarks/graphormer_rt`: GraphormerRT adapter and vendored upstream code.
- `tests`: configuration and chromatographic-input tests.

## ChromaGRT environment

Create the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate chromagrt
```

GraphormerRT uses a separate Python 3.9 environment because of its upstream
Fairseq dependencies; see `src/benchmarks/graphormer_rt/README.md`.

## Curated data release

The Zenodo release will provide the complete curated datasets required to train
ChromaGRT directly. After extracting it at the repository root, it will provide
the fixed molecular-descriptor table (`data/complete_moldesc.tsv`) and the
predefined ChromaGRT splits for the random, Bemis--Murcko,
chromatographic-condition, NPLS-filtered SMRT, and unfiltered SMRT scenarios.
Those split tables already contain the filtered and preprocessed retention-time,
chromatographic-condition, gradient, column, and Tanaka information required
by the model. The release will also include the preprocessed and final
Tanaka-enriched curated tables for the three dataset variants, plus the
versioned fold manifests.

Consequently, training from the Zenodo release requires neither downloading
RepoRT nor rerunning its filtering, preprocessing, column-information
enrichment, PubChem queries, or RDKit fallback calculations. The frozen
`complete_moldesc.tsv` is required for exact numerical reproduction because
values returned by the live PubChem service can change over time.

## Rebuild the RepoRT RP data pipeline

Generated data are not stored in Git. The pipeline fixes the RepoRT revision,
repository IDs, data variants, and fold manifests to those used in the article.
Run its single entry point from the repository root:

```bash
conda run --no-capture-output -n chromagrt \
  python -m src.RepoRT_data_processing.prepare_reportrp
```

To write the reconstructed data outside the default `data/` directory, provide
an explicit root:

```bash
conda run --no-capture-output -n chromagrt \
  python -m src.RepoRT_data_processing.prepare_reportrp \
  --data-root /path/to/data
```

It downloads RepoRT IDs 1--392, prepares the `no_SMRT`, NPLS-filtered
`with_SMRT`, and unfiltered `with_SMRT_full` variants, enriches them with
Tanaka descriptors, retrieves the molecular descriptors, and applies the
versioned `article-v1` fold manifests. It refuses to overwrite generated
files. Tanaka values are first extracted from GraphormerRT metadata and then
missing vectors are completed from the versioned article-curated stationary
phase table; this reproduces the article coverage of 117 out of 151 RP
conditions. Manifest application also reproduces the historical row order:
CC training rows are grouped by base fold, while random and BM retain source
order. The PubChem descriptor stage can take a long time; omit it only when a
compatible `data/complete_moldesc.tsv` is already available:

```bash
conda run --no-capture-output -n chromagrt \
  python -m src.RepoRT_data_processing.prepare_reportrp \
    --skip-molecular-descriptors
```

Use `--skip-download` only when the exact raw RepoRT tree has already been
placed under `data/RepoRT/raw_data` and `data/RepoRT_RP/raw_data`. The command
writes `data/RepoRT_RP/reproduction_manifest.json` with the source revision,
selected variants, and generated locations. It does not train a model.

## Inspect or run ChromaGRT

The default configuration is the complete reference model: training-fold
standardization, dropout 0.05, Tanaka descriptors, MAE optimization with AdamW
(`lr=5e-5`, `weight_decay=0.01`), validation-MAE checkpointing, early stopping
with patience 30, seed 42, and a maximum of 250 epochs. Inspect the resolved
configuration without training:

```bash
python -m src.training.RepoRT.predefined_split.main --print-config
```

Train and evaluate one predefined split:

```bash
python -m src.training.RepoRT.predefined_split.main \
  --shared-split-dir data/benchmarks/article-v1/random/chromagrt_splits/fold_0 \
  --results-dir logs/benchmarks/chromagrt_random/fold_0
```

Run all folds and forward model options to every run:

```bash
src/benchmarks/shared_folds/scripts/run_chromagrt_kfold.sh \
  data/benchmarks/article-v1/random \
  --condition-normalization standard \
  --dropout 0.05 \
  --use-tanaka
```

The complete shared-fold workflow and the input-ablation commands are documented
in `src/benchmarks/shared_folds/README.md`.

## Tests

The test suite does not train a model:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```
