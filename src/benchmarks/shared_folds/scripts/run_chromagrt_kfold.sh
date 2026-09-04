#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
INPUT_DIR="${1:-${PROJECT_ROOT}/data/benchmarks/article-v1/cc}"
if [ "$#" -gt 0 ]; then
    shift
fi
CHROMAGRT_ARGS=("$@")
RESULT_ROOT="${CHROMAGRT_SHARED_RESULTS_ROOT:-${PROJECT_ROOT}/logs/benchmarks/shared_folds/chromagrt_cc_kfold_$(date +%Y%m%d_%H%M%S)}"
CONDA_EXE="${CONDA_EXE:-conda}"

case "${INPUT_DIR}" in
    /*) ;;
    *) INPUT_DIR="${PROJECT_ROOT}/${INPUT_DIR}" ;;
esac
case "${RESULT_ROOT}" in
    /*) ;;
    *) RESULT_ROOT="${PROJECT_ROOT}/${RESULT_ROOT}" ;;
esac

if [ -d "${INPUT_DIR}/fold_0" ]; then
    SPLIT_ROOT="${INPUT_DIR}"
elif [ -d "${INPUT_DIR}/chromagrt_splits/fold_0" ]; then
    SPLIT_ROOT="${INPUT_DIR}/chromagrt_splits"
else
    echo "No fold directories found under ${INPUT_DIR} or ${INPUT_DIR}/chromagrt_splits" >&2
    exit 1
fi

shopt -s nullglob
split_dirs=()
for candidate in "${SPLIT_ROOT}"/fold_*; do
    if [ -d "${candidate}" ]; then
        split_dirs+=("${candidate}")
    fi
done
if [ "${#split_dirs[@]}" -eq 0 ]; then
    echo "No fold directories found under ${SPLIT_ROOT}" >&2
    exit 1
fi

if [ -e "${RESULT_ROOT}" ]; then
    echo "Refusing to reuse existing results directory: ${RESULT_ROOT}" >&2
    exit 1
fi
mkdir -p "${RESULT_ROOT}"
cd "${PROJECT_ROOT}"

for split_dir in "${split_dirs[@]}"; do
    fold_name="$(basename "${split_dir}")"
    echo "== ${fold_name} =="
    REPORT_SHARED_SPLIT_DIR="${split_dir}" \
    REPORT_RESULTS_DIR="${RESULT_ROOT}/${fold_name}/" \
    "${CONDA_EXE}" run --no-capture-output -n "${CHROMAGRT_CONDA_ENV:-chromagrt}" \
      python -m src.training.RepoRT.predefined_split.main "${CHROMAGRT_ARGS[@]}"
done

"${CONDA_EXE}" run --no-capture-output -n "${CHROMAGRT_CONDA_ENV:-chromagrt}" \
  python -m src.benchmarks.shared_folds.summarize_kfold_results \
    --kind chromagrt \
    --results-root "${RESULT_ROOT}"

echo "results: ${RESULT_ROOT}"
