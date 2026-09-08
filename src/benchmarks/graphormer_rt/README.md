# GraphormerRT comparison

`upstream/` contains a local copy of the original
[Graphormer-RT](https://github.com/HopkinsLaboratory/Graphormer-RT). The files
outside that directory adapt its reversed-phase workflow to the data
reconstructed by this project.

GraphormerRT is not part of the ChromaGRT environment or the ChromaGRT Zenodo
data archive. It requires a separate Python 3.9 environment with the upstream
Fairseq, DGL, and PyTorch Geometric dependencies.

## Article evaluation: fixed 10-fold benchmarks

The article compares ChromaGRT and GraphormerRT on the fixed `random`,
`bm_scaffold`, and `cc` folds. These GraphormerRT assets are not included in
the ChromaGRT Zenodo archive, so reproduce the complete local data pipeline
first:

```bash
conda run --no-capture-output -n chromagrt \
  python -m src.RepoRT_data_processing.prepare_reportrp
```

This applies the versioned article fold manifests and creates the shared
scenario assets under `data/benchmarks/article-v1/`. It generates a
GraphormerRT-compatible `report_rp.csv`, a base `RP_metadata.pickle`, and the
per-fold train/validation/test manifests without discarding observations to
match a pre-existing GraphormerRT CSV.

For one scenario, construct the fold-specific metadata and then run the ten
GraphormerRT folds from the repository root:

```bash
conda run --no-capture-output -n chromagrt \
  python -m src.benchmarks.graphormer_rt.data.build_fold_specific_metadata \
    --asset-dir data/benchmarks/article-v1/cc

bash src/benchmarks/graphormer_rt/scripts/run_graphormer_rt_kfold_fold_imputed.sh \
  data/benchmarks/article-v1/cc
```

Replace `cc` with `random` or `bm_scaffold` for the other article scenarios.
The metadata builder writes one `RP_metadata.pickle` per fold. The launcher uses
the resulting fold-specific metadata for both training and evaluation. It
trains one model per fold, and writes aggregate metrics to its result directory.

Shared-fold definitions and the protocol for comparing another model on the
same test observations are documented in
[the shared-fold benchmark README](../shared_folds/README.md).
