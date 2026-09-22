#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${BENCH_DIR}/../../.." && pwd)"
UPSTREAM_DIR="${BENCH_DIR}/upstream"
DATA_DIR="${GRAPHORMER_RT_DATA_DIR:-${PROJECT_ROOT}/data/benchmarks/graphormer_rt/report_rp_paper}"
SAVE_DIR="${GRAPHORMER_RT_SAVE_DIR:-${PROJECT_ROOT}/logs/benchmarks/graphormer_rt/report_rp_paper}"

case "${DATA_DIR}" in
    /*) ;;
    *) DATA_DIR="${PROJECT_ROOT}/${DATA_DIR}" ;;
esac

case "${SAVE_DIR}" in
    /*) ;;
    *) SAVE_DIR="${PROJECT_ROOT}/${SAVE_DIR}" ;;
esac

export GRAPHORMER_RT_RP_CSV="${GRAPHORMER_RT_RP_CSV:-${DATA_DIR}/report_rp.csv}"
export GRAPHORMER_RT_RP_METADATA="${GRAPHORMER_RT_RP_METADATA:-${DATA_DIR}/RP_metadata.pickle}"
export GRAPHORMER_RT_TRAIN_FRACTION="${GRAPHORMER_RT_TRAIN_FRACTION:-0.8}"
export GRAPHORMER_RT_VAL_FRACTION="${GRAPHORMER_RT_VAL_FRACTION:-0.1}"
export GRAPHORMER_RT_SPLIT_MODE="${GRAPHORMER_RT_SPLIT_MODE:-random}"
export GRAPHORMER_RT_SPLIT_SEED="${GRAPHORMER_RT_SPLIT_SEED:-23}"

mkdir -p "${SAVE_DIR}"
cd "${UPSTREAM_DIR}/examples/property_prediction"

PYTHON_CUDA_LIB_PATHS="$(
python - <<'PY'
import glob
import os
import site

paths = []
for root in site.getsitepackages():
    paths.extend(glob.glob(os.path.join(root, "nvidia", "*", "lib")))

for path in paths:
    if os.path.isdir(path):
        print(path)
PY
)"
if [ -n "${PYTHON_CUDA_LIB_PATHS}" ]; then
    export LD_LIBRARY_PATH="$(printf '%s\n' "${PYTHON_CUDA_LIB_PATHS}" | paste -sd: -)${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export PYTHONPATH="${UPSTREAM_DIR}/fairseq:${UPSTREAM_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python -m fairseq_cli.train \
    --user-dir ../../graphormer \
    --batch-size 64 \
    --num-workers "${GRAPHORMER_RT_NUM_WORKERS:-16}" \
    --ddp-backend=legacy_ddp \
    --seed 23 \
    --user-data-dir report_rp_training_dataset \
    --dataset-name RT_Library \
    --task graph_prediction_with_flag \
    --criterion "${GRAPHORMER_RT_CRITERION:-mse_rt}" \
    --arch graphormer_base \
    --num-classes 1 \
    --attention-dropout 0.15 --act-dropout 0.10 --dropout 0.10 \
    --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
    --lr-scheduler polynomial_decay --power 1 --warmup-updates 33281 --total-num-update 221875 \
    --scale-to-max-epoch --warmup-ratio 0.15 \
    --lr "${GRAPHORMER_RT_LR:-1e-4}" \
    --fp16 \
    --encoder-layers 8 \
    --encoder-embed-dim 512 \
    --encoder-ffn-embed-dim "${GRAPHORMER_RT_ENCODER_FFN_EMBED_DIM:-512}" \
    --encoder-attention-heads 64 \
    --fp16-scale-tolerance 0.05 \
    --mlp-layers 5 \
    --max-epoch "${GRAPHORMER_RT_MAX_EPOCH:-250}" \
    --patience "${GRAPHORMER_RT_PATIENCE:-30}" \
    --patience-after-warmup \
    --no-epoch-checkpoints \
    --freeze-level 0 \
    --save-dir "${SAVE_DIR}"

if [ "${GRAPHORMER_RT_EVAL_AFTER_TRAIN:-1}" = "1" ]; then
    cd "${PROJECT_ROOT}"
    IFS=',' read -r -a EVAL_SPLITS <<< "${GRAPHORMER_RT_EVAL_SPLITS:-valid,test}"
    for split in "${EVAL_SPLITS[@]}"; do
        python -m src.benchmarks.graphormer_rt.evaluate_report_rp \
            --checkpoint "${SAVE_DIR}/checkpoint_best.pt" \
            --split "${split}" \
            --output-dir "${SAVE_DIR}/eval_${split}" \
            --data-dir "${DATA_DIR}" \
            --train-fraction "${GRAPHORMER_RT_TRAIN_FRACTION}" \
            --val-fraction "${GRAPHORMER_RT_VAL_FRACTION}" \
            --split-mode "${GRAPHORMER_RT_SPLIT_MODE}" \
            --split-seed "${GRAPHORMER_RT_SPLIT_SEED}" \
            --criterion "${GRAPHORMER_RT_CRITERION:-mse_rt}" \
            --encoder-layers 8 \
            --encoder-embed-dim 512 \
            --encoder-ffn-embed-dim "${GRAPHORMER_RT_ENCODER_FFN_EMBED_DIM:-512}" \
            --encoder-attention-heads 64 \
            --mlp-layers 5 \
            --num-workers "${GRAPHORMER_RT_NUM_WORKERS:-16}" \
            --batch-size 64 \
            --seed 23
    done
fi
