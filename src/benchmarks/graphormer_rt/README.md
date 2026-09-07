# GraphormerRT comparison

`upstream/` contains a local copy of the original
[Graphormer-RT](https://github.com/HopkinsLaboratory/Graphormer-RT). The files
outside that directory adapt its RP workflow to the data reconstructed by this
project.

GraphormerRT is not part of the ChromaGRT environment or Zenodo data archive.
It requires a separate Python 3.9 environment with the upstream Fairseq, DGL,
and PyTorch Geometric dependencies.

## Reconstruct the RP inputs

After preparing the local RepoRT data, run:

```bash
conda run --no-capture-output -n chromagrt \
  python -m src.benchmarks.graphormer_rt.data.build_report_rp_inputs
```

This creates:

```text
data/benchmarks/graphormer_rt/report_rp_paper/report_rp.csv
data/benchmarks/graphormer_rt/report_rp_paper/RP_metadata.pickle
```

The builder uses the RP raw retention-time table, the preprocessed condition
and gradient tables, and the upstream sample metadata. It applies the local
implementation of the paper and upstream gradient filters. Tanaka/HSMB values
are recovered where possible; remaining fields are set to zero.

## Train and evaluate

Run one randomized 80/10/10 split in the GraphormerRT environment:

```bash
conda run --no-capture-output -n chromagrt_graphormer_rt \
  bash src/benchmarks/graphormer_rt/scripts/run_graphormer_rt.sh
```

The wrapper writes checkpoints and evaluation results under
`logs/benchmarks/graphormer_rt/report_rp_paper` by default. It uses the
`mse_rt` criterion unless `GRAPHORMER_RT_CRITERION` is set. Shared-fold runs
are described in `src/benchmarks/shared_folds/README.md`.

## Article 10-fold validation

The article uses fixed random, Bemis--Murcko, and chromatographic-condition
folds. These GraphormerRT assets are not included in the ChromaGRT Zenodo data
archive. Rebuild the complete local pipeline first:

```bash
conda run --no-capture-output -n chromagrt \
  python -m src.RepoRT_data_processing.prepare_reportrp
```

Then run the ten folds for one scenario from the repository root:

```bash
src/benchmarks/shared_folds/scripts/run_graphormer_rt_kfold.sh \
  data/benchmarks/article-v1/cc
```

Replace `cc` with `random` or `bm_scaffold` for the other article scenarios.
The launcher uses the `chromagrt_graphormer_rt` environment by default, trains
one model per fold, and writes the aggregate metrics to its result directory.
This is separate from the randomized 80/10/10 run described above.
