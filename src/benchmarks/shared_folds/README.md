# Shared-fold benchmarks

This directory contains the utilities used to train and evaluate ChromaGRT on
the fixed folds reported in the article. The exact assignments are stored in
`fold_manifests/article-v1`; the Zenodo data release provides the corresponding
prebuilt ChromaGRT split tables and frozen molecular descriptors.

## Article benchmarks

After extracting the Zenodo archive at the repository root, each scenario is
ready to run. The split tables already contain the curated retention-time,
chromatographic-condition, gradient, column, and Tanaka inputs. ChromaGRT also
reads `data/complete_moldesc.tsv` by default.

Run the reference configuration on the random folds:

```bash
src/benchmarks/shared_folds/scripts/run_chromagrt_kfold.sh \
  data/benchmarks/article-v1/random
```

The other no-SMRT article scenarios are available under
`data/benchmarks/article-v1/bm_scaffold` and
`data/benchmarks/article-v1/cc`. The CC configuration reported in the article
uses deterministic condition normalization, dropout 0.15, and no Tanaka
descriptors:

```bash
src/benchmarks/shared_folds/scripts/run_chromagrt_kfold.sh \
  data/benchmarks/article-v1/cc \
  --condition-normalization deterministic --dropout 0.15 --no-use-tanaka
```

The two SMRT analyses use the random folds below. They evaluate ChromaGRT only.

```text
data/benchmarks/article-v1/smrt_filtered_random
data/benchmarks/article-v1/smrt_unfiltered_random
```

The launcher accepts either an asset directory such as those above or its
`chromagrt_splits` subdirectory. It creates a new result directory, trains the
ten folds, and writes a summary when all runs finish. The default model is the
reference configuration: standard training-fold normalization, dropout 0.05,
Tanaka descriptors, AdamW (`lr=5e-5`, `weight_decay=0.01`), MAE monitoring,
patience 30, seed 42, and a maximum of 250 epochs.

### Input ablations

All input ablations use the same random split directory:

```bash
SPLITS=data/benchmarks/article-v1/random/chromagrt_splits
```

Pass `--exclude-condition-blocks composition`, `static`, or `gradient` to omit
one chromatographic block. Pass
`--exclude-molecular-descriptors mono_iso_mass,xlogp` to omit both molecular
descriptors. For example:

```bash
CHROMAGRT_SHARED_RESULTS_ROOT=logs/benchmarks/no_composition \
  src/benchmarks/shared_folds/scripts/run_chromagrt_kfold.sh "$SPLITS" \
  --exclude-condition-blocks composition
```

To summarize an existing ChromaGRT run manually:

```bash
conda run --no-capture-output -n chromagrt \
  python -m src.benchmarks.shared_folds.summarize_kfold_results \
    --kind chromagrt --results-root /path/to/results
```

## GraphormerRT comparison

GraphormerRT requires its own Python 3.9 environment and is not included in
the ChromaGRT Zenodo data archive. Its setup and local reconstruction workflow
are documented in `src/benchmarks/graphormer_rt/README.md`; see also the
[upstream Graphormer-RT README](https://github.com/HopkinsLaboratory/Graphormer-RT#readme).
Once compatible GraphormerRT assets have been created, run its fold launcher
with the asset directory, for example:

```bash
src/benchmarks/shared_folds/scripts/run_graphormer_rt_kfold.sh \
  data/benchmarks/article-v1/cc
```

## Rebuilding or refreshing folds

The article pipeline does not recalculate its fold assignments. To reconstruct
the curated data from RepoRT and apply the fixed manifests, use
`src.RepoRT_data_processing.prepare_reportrp`, as described in the repository
root README.

The following scripts are retained for a future RepoRT refresh, when new fold
assignments and comparison assets may be needed. They are not part of the
article-reproduction workflow:

- `generate_chromagrt_folds.py` creates candidate CC, scaffold, or random
  within-condition folds from a refreshed processed dataset.
- `build_kfold_assets_from_chromagrt_folds.py` converts such folds into matching
  ChromaGRT and GraphormerRT assets.
- `export_fold_manifests.py` exports compact, versionable assignments from three
  `master_manifest.csv` files.
- `build_random_or_bm_kfold_assets.py` creates exploratory random or
  Bemis--Murcko assets from a supplied dataset.

`build_assets_from_fold_manifest.py` is the manifest-application step used by
the RepoRT reconstruction pipeline. It applies a fixed assignment rather than
calculating a new one.
