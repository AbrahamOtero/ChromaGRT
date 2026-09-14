# Reproducible shared-fold benchmarks

This directory defines a reusable, model-agnostic benchmark protocol for
retention-time prediction. Its purpose is to enable researchers to train and
evaluate different models on exactly the same data partitions, so that their
reported metrics are directly comparable.

The article uses this protocol to compare ChromaGRT with GraphormerRT. That is
the reference application supplied by this repository, not the sole purpose of
the folds. Any researcher can apply the same fixed assignments to a compatible
model and report results on the same test observations.

## The benchmark artefacts

The exact, versioned article assignments are in
[`fold_manifests/article-v1`](../../../fold_manifests/article-v1). Each manifest row identifies an
observation through `cc_id`, `molecule_id`, and `inchi.std`, and assigns it a
`base_fold`.

The three scenarios are:

```text
random       Random partition of the curated observations.
bm_scaffold  Bemis--Murcko scaffold partition.
cc           Chromatographic-condition partition.
```

For a ten-fold run `k`:

- `base_fold == k` is the test set;
- `base_fold == (k + 1) mod 10` is the validation set; and
- all other rows are the training set.

For `bm_scaffold`, rows with `base_fold == -1` have no scaffold and always
remain in the training set. See the
[fold-manifest documentation](../../../fold_manifests/README.md) for the
complete manifest format and the requirements for applying it to a dataset.

Do not recalculate these folds when benchmarking another model. Reuse the
published manifest unchanged and preserve the membership of every train,
validation, and test split. If a model cannot process an observation, report
that fact and the excluded identifiers explicitly: a metric computed on a
different test cohort is not directly comparable to the article results.

## Reporting comparable metrics

Report at least MAE, RMSE, MRE, and the number of evaluated test observations for
each fold. The article summaries use the number of test observations as the
weight: MAE and MRE are weighted means, and global RMSE is computed as
`sqrt(sum(n_i * RMSE_i^2) / sum(n_i))`. Reporting the per-fold metrics and
sample counts makes it possible to reproduce the aggregate result even when
folds have different sizes.

The included summarizer reads the result layouts written by ChromaGRT and
GraphormerRT; use `--kind chromagrt` or `--kind graphormer_rt`, respectively.
Other implementations can use the same aggregation rule or adapt the
summarizer to their own result files.

## Use the published folds with another model

The recommended route is to download the curated data release from Zenodo and
use its scenario-level `master_manifest.csv` together with the corresponding
fold manifest. The master manifest contains the curated data and their
assigned fold, while the compact manifest remains the independently versioned
source of truth for the assignment.

To prepare inputs for another model:

1. Select `random`, `bm_scaffold`, or `cc`, and use the matching `article-v1`
   manifest.
2. Map the model's input data to the manifest using the three observation keys
   (`cc_id`, `molecule_id`, and `inchi.std`).
3. Construct each run's train, validation, and test data with the split rule
   stated above, without changing the assigned rows.
4. Train and evaluate the model on all ten folds, then report per-fold and
   aggregate metrics as described above.

If the curated data have been rebuilt locally instead, apply the fixed manifest
with `src.benchmarks.shared_folds.build_assets_from_fold_manifest`; it validates
that the reconstructed dataset and the manifest identify exactly the same
observations. The repository-root README documents the reconstruction pipeline.

## ChromaGRT

After extracting the Zenodo archive at the repository root, the prebuilt
ChromaGRT split tables are available under `data/benchmarks/article-v1`. They
already contain the curated retention-time, chromatographic-condition,
gradient, column, and Tanaka inputs. ChromaGRT reads
`data/complete_moldesc.tsv` by default.

Run the reference configuration on the random folds:

```bash
src/benchmarks/shared_folds/scripts/run_chromagrt_kfold.sh \
  data/benchmarks/article-v1/random
```

The other scenarios are `data/benchmarks/article-v1/bm_scaffold` and
`data/benchmarks/article-v1/cc`. The CC configuration reported in the article
uses deterministic condition normalization, dropout 0.15, and no Tanaka
descriptors:

```bash
src/benchmarks/shared_folds/scripts/run_chromagrt_kfold.sh \
  data/benchmarks/article-v1/cc \
  --condition-normalization deterministic --dropout 0.15 --no-use-tanaka
```

The launcher trains the ten folds and writes a summary
when all runs finish. The default configuration uses training-fold
normalization, dropout 0.05, Tanaka descriptors, AdamW (`lr=5e-5`,
`weight_decay=0.01`), MAE monitoring, patience 30, seed 42, and a maximum of
250 epochs.

### Input ablations

Pass `--exclude-condition-blocks composition`, `static`, or `gradient` to omit
one chromatographic block. Pass
`--exclude-molecular-descriptors mono_iso_mass,xlogp` to omit both molecular
descriptors. For example:

```bash
SPLITS=data/benchmarks/article-v1/random/chromagrt_splits
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


## GraphormerRT

GraphormerRT  uses the same folds as ChromaGRT. GraphormerRT requires its own Python 3.9 environment and is not included in
the ChromaGRT Zenodo data archive. Its setup, local data reconstruction, and
asset-preparation workflow are documented in
[`src/benchmarks/graphormer_rt/README.md`](../graphormer_rt/README.md); see
also the [upstream Graphormer-RT README](https://github.com/HopkinsLaboratory/Graphormer-RT#readme).
Once compatible GraphormerRT assets have been prepared, run a scenario with:

```bash
bash src/benchmarks/graphormer_rt/scripts/run_graphormer_rt_kfold_fold_imputed.sh \
  data/benchmarks/article-v1/cc
```

## Creating a future benchmark release

The article-reproduction workflow applies the fixed `article-v1` assignments.
The scripts below can be used for a future
curated-data refresh or a new benchmark release. A new release should preserve
its manifest files so that subsequent models can be evaluated on the same
partitions.

- `generate_chromagrt_folds.py` creates candidate CC, scaffold, or random
  within-condition folds from a refreshed processed dataset.
- `build_kfold_assets_from_chromagrt_folds.py` builds matching ChromaGRT and
  GraphormerRT assets from a new fold set.
- `export_fold_manifests.py` exports compact, versioned assignments from three
  full `master_manifest.csv` files.

`build_assets_from_fold_manifest.py` is the manifest-application step used by
the local RepoRT reconstruction pipeline. It applies a fixed assignment rather
than calculating a new one.
