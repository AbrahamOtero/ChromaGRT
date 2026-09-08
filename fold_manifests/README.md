# Article fold manifests

`article-v1` contains the exact base-fold assignments used for the curated
RepoRT reversed-phase no-SMRT dataset with Tanaka descriptors.

These manifests are intended to be reused unchanged by any compatible
retention-time model, so that independently reported metrics use the same
training, validation, and test observations as the article benchmarks.

Each TSV has one row per observation and these columns:

```text
cc_id    molecule_id    inchi.std    base_fold
```

The three scenarios are `random`, `bm_scaffold`, and `cc`. For run `k`, rows
with `base_fold == k` form the test set; rows with
`base_fold == (k + 1) mod 10` form the validation set; all remaining rows form
the training set. In the Bemis--Murcko scenario, `base_fold == -1` (no scaffold) is always in
training.

The curated dataset is distributed separately. It must preserve cc_id, molecule_id and inchi.std
so that these assignments can be applied. `metadata.json` records
the dataset release label, record counts, and splits.
`cc_id_to_dir_id.tsv` provides the corresponding original RepoRT directory
identifier for each curated chromatographic condition.

To export a new manifest release from three full `master_manifest.csv` files run:

```bash
conda run --no-capture-output -n chromagrt \
  python -m src.benchmarks.shared_folds.export_fold_manifests \
  --random-master /path/to/random/master_manifest.csv \
  --bm-master /path/to/bm_scaffold/master_manifest.csv \
  --cc-master /path/to/cc/master_manifest.csv \
  --output-dir fold_manifests/article-v2 \
  --dataset-version article-v2
```

To rebuild the training assets for one scenario from the
curated data, apply the manifest rather than recalculating folds:

```bash
conda run --no-capture-output -n chromagrt \
  python -m src.benchmarks.shared_folds.build_assets_from_fold_manifest \
  --scenario random \
  --input-path /path/to/complete_processed_data_with_tanaka.tsv \
  --fold-manifest fold_manifests/article-v1/random_10fold.tsv \
  --preprocessed-cc /path/to/preprocessed_cc_data.tsv \
  --preprocessed-grad /path/to/preprocessed_gradient_data.tsv \
  --column-properties /path/to/column_properties_from_upstream_sample.tsv \
  --output-dir data/benchmarks/article-v1/random
```

The resulting directory contains the ten ChromaGRT split directories, metadata,
and per-fold manifests. Its generated split tables and master manifest retain
the `cc_id`, `molecule_id`, and `inchi.std` identifiers for every observation.
The command fails if any
dataset row is absent from the fold manifest, or vice versa.
