#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
ASSET_DIR="${1:-${PROJECT_ROOT}/data/benchmarks/shared_folds/report_rp_no_smrt_cc_kfold}"
case "${ASSET_DIR}" in
    /*) ;;
    *) ASSET_DIR="${PROJECT_ROOT}/${ASSET_DIR}" ;;
esac

if [ -f "${ASSET_DIR}/report_rp.csv" ] && [ -f "${ASSET_DIR}/RP_metadata.pickle" ]; then
    DEFAULT_DATA_VIEW="${ASSET_DIR}"
else
    DEFAULT_DATA_VIEW="${PROJECT_ROOT}/data/benchmarks/graphormer_rt/report_rp_paper_grad_minutes_graphormer_rt_gradient"
fi
DATA_VIEW="${GRAPHORMER_RT_SHARED_DATA_VIEW:-${DEFAULT_DATA_VIEW}}"
RESULT_ROOT="${GRAPHORMER_RT_SHARED_RESULTS_ROOT:-${PROJECT_ROOT}/logs/benchmarks/shared_folds/graphormer_rt_cc_kfold_$(date +%Y%m%d_%H%M%S)}"
CONDA_EXE="${CONDA_EXE:-conda}"
case "${DATA_VIEW}" in
    /*) ;;
    *) DATA_VIEW="${PROJECT_ROOT}/${DATA_VIEW}" ;;
esac
case "${RESULT_ROOT}" in
    /*) ;;
    *) RESULT_ROOT="${PROJECT_ROOT}/${RESULT_ROOT}" ;;
esac

shopt -s nullglob
manifests=("${ASSET_DIR}"/graphormer_rt_manifests/fold_*_graphormer_rt_manifest.csv)
if [ "${#manifests[@]}" -eq 0 ]; then
    echo "No fold manifests found under ${ASSET_DIR}/graphormer_rt_manifests" >&2
    exit 1
fi

if [ -e "${RESULT_ROOT}" ]; then
    echo "Refusing to reuse existing results directory: ${RESULT_ROOT}" >&2
    exit 1
fi
mkdir -p "${RESULT_ROOT}"

for manifest in "${manifests[@]}"; do
    fold_name="$(basename "${manifest}" _graphormer_rt_manifest.csv)"
    echo "== ${fold_name} =="
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}" \
    GRAPHORMER_RT_NUM_WORKERS="${GRAPHORMER_RT_NUM_WORKERS:-16}" \
    GRAPHORMER_RT_DATA_DIR="${DATA_VIEW}" \
    GRAPHORMER_RT_MANIFEST="${manifest}" \
    GRAPHORMER_RT_ALLOW_MANIFEST_SUBSET=1 \
    GRAPHORMER_RT_SHUFFLE_TRAIN_MANIFEST="${GRAPHORMER_RT_SHUFFLE_TRAIN_MANIFEST:-1}" \
    GRAPHORMER_RT_SAVE_DIR="${RESULT_ROOT}/${fold_name}" \
    GRAPHORMER_RT_CRITERION="${GRAPHORMER_RT_CRITERION:-mse_rt}" \
    GRAPHORMER_RT_ENCODER_FFN_EMBED_DIM="${GRAPHORMER_RT_ENCODER_FFN_EMBED_DIM:-512}" \
    GRAPHORMER_RT_MAX_EPOCH="${GRAPHORMER_RT_MAX_EPOCH:-250}" \
    GRAPHORMER_RT_PATIENCE="${GRAPHORMER_RT_PATIENCE:-30}" \
    "${CONDA_EXE}" run --no-capture-output -n "${GRAPHORMER_RT_CONDA_ENV:-chromagrt_graphormer_rt}" \
      bash "${PROJECT_ROOT}/src/benchmarks/graphormer_rt/scripts/run_graphormer_rt.sh"
done

"${CONDA_EXE}" run --no-capture-output -n "${GRAPHORMER_RT_SUMMARY_CONDA_ENV:-chromagrt}" \
  python -m src.benchmarks.shared_folds.summarize_kfold_results \
    --kind graphormer_rt \
    --results-root "${RESULT_ROOT}"

echo "results: ${RESULT_ROOT}"
