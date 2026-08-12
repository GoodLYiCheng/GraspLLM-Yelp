#!/usr/bin/env bash
# Run the complete static YelpZip evaluation grid in one reproducible command.
set -euo pipefail

MODEL_PATH=""; MODEL_BASE=""; DATASET_ROOT=""; SUPPORT_ROOT=""; OUTPUT_DIR=""; GPUS="0,1"; ICL_ONLY=false
MAX_VALIDATION_QUERIES=""; VALIDATION_SUBSAMPLE_SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2;;
        --model-base) MODEL_BASE="$2"; shift 2;;
        --dataset-root) DATASET_ROOT="$2"; shift 2;;
        --support-root) SUPPORT_ROOT="$2"; shift 2;;
        --output-dir) OUTPUT_DIR="$2"; shift 2;;
        --gpus) GPUS="$2"; shift 2;;
        --max-validation-queries) MAX_VALIDATION_QUERIES="$2"; shift 2;;
        --validation-subsample-seed) VALIDATION_SUBSAMPLE_SEED="$2"; shift 2;;
        --icl-only) ICL_ONLY=true; shift;;
        -h|--help)
            echo "Usage: $0 --model-path CKPT --model-base MODEL --support-root DIR [--dataset-root DIR] [--output-dir DIR] [--gpus 0,1] [--max-validation-queries 1000] [--validation-subsample-seed 42] [--icl-only]"; exit 0;;
        *) echo "unknown argument: $1" >&2; exit 1;;
    esac
done

[[ -n "$MODEL_PATH" ]] || { echo "--model-path is required" >&2; exit 1; }
[[ -n "$MODEL_BASE" ]] || { echo "--model-base is required" >&2; exit 1; }
[[ -n "$SUPPORT_ROOT" ]] || { echo "--support-root is required (existing 1/5/10-shot support directory)" >&2; exit 1; }

REPO=$(cd "$(dirname "$0")/.." && pwd)
DATASET_ROOT=${DATASET_ROOT:-${GRASPLLM_DATASET_ROOT:-$REPO/dataset}}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/artifacts/yelpzip_full_evaluation}
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export GRASPLLM_DATASET_ROOT="$DATASET_ROOT"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

IFS=',' read -r -a GPU_IDS <<< "$GPUS"
GPU_RUR=${GPU_IDS[0]}
GPU_RBR=${GPU_IDS[1]:-${GPU_IDS[0]}}
RELATIONS=(yelpzip_rur yelpzip_rbr)
SHOTS=(1 5 10)
SEEDS=(42 43 44 45 46)
VALIDATION_ARGS=()
if [[ -n "$MAX_VALIDATION_QUERIES" ]]; then
    [[ "$MAX_VALIDATION_QUERIES" =~ ^[1-9][0-9]*$ ]] || { echo "--max-validation-queries must be a positive integer" >&2; exit 1; }
    VALIDATION_ARGS+=(--max-validation-queries "$MAX_VALIDATION_QUERIES"
                      --validation-subsample-seed "$VALIDATION_SUBSAMPLE_SEED")
fi

run_eval() {
    local gpu="$1" relation="$2" output="$3"
    shift 3
    CUDA_VISIBLE_DEVICES="$gpu" python -u "$REPO/eval/eval_yelp_probability.py" \
        --model-path "$MODEL_PATH" --model-base "$MODEL_BASE" --dataset "$relation" \
        --output-dir "$output" --device cuda --max-length 4096 --query-max-tokens 512 \
        "${VALIDATION_ARGS[@]}" "$@"
}

run_relation_pair() {
    local rur_output="$1" rbr_output="$2"
    shift 2
    if [[ "$GPU_RUR" == "$GPU_RBR" ]]; then
        run_eval "$GPU_RUR" yelpzip_rur "$rur_output" "$@"
        run_eval "$GPU_RBR" yelpzip_rbr "$rbr_output" "$@"
    else
        run_eval "$GPU_RUR" yelpzip_rur "$rur_output" "$@" & local rur_pid=$!
        run_eval "$GPU_RBR" yelpzip_rbr "$rbr_output" "$@" & local rbr_pid=$!
        wait "$rur_pid"
        wait "$rbr_pid"
    fi
}

if [[ "$ICL_ONLY" == false ]]; then
    echo "[all-eval] GPU: zero-shot RUR/RBR"
    run_relation_pair "$OUTPUT_DIR/zero_shot/yelpzip_rur" "$OUTPUT_DIR/zero_shot/yelpzip_rbr"
else
    echo "[all-eval] skipping zero-shot (--icl-only)"
fi

for shot in "${SHOTS[@]}"; do
    case "$shot" in
        1) SUPPORT_TEXT_TOKENS=96; QUERY_TEXT_TOKENS=512;;
        5) SUPPORT_TEXT_TOKENS=64; QUERY_TEXT_TOKENS=384;;
        10) SUPPORT_TEXT_TOKENS=32; QUERY_TEXT_TOKENS=256;;
    esac
    for seed in "${SEEDS[@]}"; do
        echo "[all-eval] GPU: ICL k=$shot seed=$seed"
        # RUR and RBR need relation-specific support files, so run explicitly.
        if [[ "$GPU_RUR" == "$GPU_RBR" ]]; then
            run_eval "$GPU_RUR" yelpzip_rur "$OUTPUT_DIR/few_shot/yelpzip_rur/k${shot}_seed${seed}" \
                --validation-jsonl "$SUPPORT_ROOT/yelpzip_rur/k${shot}_seed${seed}/validation_holdout.jsonl" \
                --icl-support-jsonl "$SUPPORT_ROOT/yelpzip_rur/k${shot}_seed${seed}/support_train.jsonl" \
                --support-manifest "$SUPPORT_ROOT/yelpzip_rur/k${shot}_seed${seed}/support_manifest.json" \
                --icl-support-max-tokens "$SUPPORT_TEXT_TOKENS" --query-max-tokens "$QUERY_TEXT_TOKENS" --icl-support-graphs
            run_eval "$GPU_RBR" yelpzip_rbr "$OUTPUT_DIR/few_shot/yelpzip_rbr/k${shot}_seed${seed}" \
                --validation-jsonl "$SUPPORT_ROOT/yelpzip_rbr/k${shot}_seed${seed}/validation_holdout.jsonl" \
                --icl-support-jsonl "$SUPPORT_ROOT/yelpzip_rbr/k${shot}_seed${seed}/support_train.jsonl" \
                --support-manifest "$SUPPORT_ROOT/yelpzip_rbr/k${shot}_seed${seed}/support_manifest.json" \
                --icl-support-max-tokens "$SUPPORT_TEXT_TOKENS" --query-max-tokens "$QUERY_TEXT_TOKENS" --icl-support-graphs
        else
            run_eval "$GPU_RUR" yelpzip_rur "$OUTPUT_DIR/few_shot/yelpzip_rur/k${shot}_seed${seed}" \
                --validation-jsonl "$SUPPORT_ROOT/yelpzip_rur/k${shot}_seed${seed}/validation_holdout.jsonl" \
                --icl-support-jsonl "$SUPPORT_ROOT/yelpzip_rur/k${shot}_seed${seed}/support_train.jsonl" \
                --support-manifest "$SUPPORT_ROOT/yelpzip_rur/k${shot}_seed${seed}/support_manifest.json" \
                --icl-support-max-tokens "$SUPPORT_TEXT_TOKENS" --query-max-tokens "$QUERY_TEXT_TOKENS" --icl-support-graphs & local_rur_pid=$!
            run_eval "$GPU_RBR" yelpzip_rbr "$OUTPUT_DIR/few_shot/yelpzip_rbr/k${shot}_seed${seed}" \
                --validation-jsonl "$SUPPORT_ROOT/yelpzip_rbr/k${shot}_seed${seed}/validation_holdout.jsonl" \
                --icl-support-jsonl "$SUPPORT_ROOT/yelpzip_rbr/k${shot}_seed${seed}/support_train.jsonl" \
                --support-manifest "$SUPPORT_ROOT/yelpzip_rbr/k${shot}_seed${seed}/support_manifest.json" \
                --icl-support-max-tokens "$SUPPORT_TEXT_TOKENS" --query-max-tokens "$QUERY_TEXT_TOKENS" --icl-support-graphs & local_rbr_pid=$!
            wait "$local_rur_pid"
            wait "$local_rbr_pid"
        fi
    done
done

python -m experiments.yelpzip_fewshot.summarize --root "$OUTPUT_DIR"
echo "[all-eval] complete: $OUTPUT_DIR/all_results_summary.json"
