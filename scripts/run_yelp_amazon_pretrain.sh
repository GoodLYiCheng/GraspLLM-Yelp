#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:-}
[[ -n "$STAGE" ]] || {
  echo "usage: $0 prepare|embed|stage1|stage2|stage3|support|eval|all" >&2
  exit 2
}

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

RAW_REVIEW_ROOT=${RAW_REVIEW_ROOT:?'set RAW_REVIEW_ROOT to the extracted raw_data directory'}
export GRASPLLM_DATASET_ROOT=${GRASPLLM_DATASET_ROOT:-$REPO/dataset}
export GRASPLLM_MODELS_ROOT=${GRASPLLM_MODELS_ROOT:-$REPO/models}
export GRASPLLM_CHECKPOINT_ROOT=${GRASPLLM_CHECKPOINT_ROOT:-$REPO/graspllm_checkpoints}
export QWEN3_EMB_MODEL=${GRASPLLM_QWEN3_EMB_MODEL:-${QWEN3_EMB_MODEL:-$GRASPLLM_MODELS_ROOT/Qwen3-Embedding-8B}}
BASE_MODEL=${BASE_MODEL:-$GRASPLLM_MODELS_ROOT/Qwen3-8B}
EMBED_GPUS=${EMBED_GPUS:-0,1,2,3}
STAGE1_GPU=${STAGE1_GPU:-0}
STAGE2_GPUS=${STAGE2_GPUS:-0,1,2,3}
STAGE2_LOGICAL_GPUS=${STAGE2_LOGICAL_GPUS:-0,1,2,3}
STAGE3_GPUS=${STAGE3_GPUS:-0,1,2,3}
PROJECTOR_NAME=${PROJECTOR_NAME:-grasp-qwen3-yelp-amazon-projector}
PROJECTOR_PATH="$GRASPLLM_CHECKPOINT_ROOT/$PROJECTOR_NAME"
SUPPORT_ROOT=${SUPPORT_ROOT:-$REPO/artifacts/yelpzip_yelp_amazon_support}
RESULT_ROOT=${RESULT_ROOT:-$REPO/artifacts/yelpzip_yelp_amazon_results}

BASE_EMBED_DATASETS=(
  yelpzip_rur
  amazon_cellphones_rur amazon_clothing_rur amazon_electronics_rur
  amazon_home_rur amazon_sports_rur amazon_toys_rur
)
ALL_DATASETS=(
  yelpzip_rur yelpzip_rbr
  amazon_cellphones_rur amazon_cellphones_rpr
  amazon_clothing_rur amazon_clothing_rpr
  amazon_electronics_rur amazon_electronics_rpr
  amazon_home_rur amazon_home_rpr
  amazon_sports_rur amazon_sports_rpr
  amazon_toys_rur amazon_toys_rpr
)
SOURCE_SPEC=$(IFS=-; echo "${ALL_DATASETS[*]}")

require_file() {
  [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 2; }
}

run_prepare() {
  require_file "$RAW_REVIEW_ROOT/Yelp-Dataset/yelpzip.csv"
  local args=()
  [[ "${FORCE:-0}" == "1" ]] && args+=(--overwrite)
  python -u -m preprocess.prepare_review_pretrain \
    --raw-root "$RAW_REVIEW_ROOT" \
    --dataset-root "$GRASPLLM_DATASET_ROOT" \
    --amazon-rows-per-category 100000 \
    --max-neighbors 32 --yelp-val-size 10000 --yelp-test-size 20000 --seed 42 \
    "${args[@]}"
  python -u scripts/validate_yelp_amazon_pretrain.py \
    --dataset-root "$GRASPLLM_DATASET_ROOT" --stage prepared \
    --output "$GRASPLLM_DATASET_ROOT/yelp_amazon_prepared_audit.json"
}

run_embed() {
  [[ -d "$QWEN3_EMB_MODEL" ]] || { echo "missing embedding model: $QWEN3_EMB_MODEL" >&2; exit 2; }
  for dataset in "${BASE_EMBED_DATASETS[@]}"; do
    require_file "$GRASPLLM_DATASET_ROOT/$dataset/processed_data.pt"
    bash scripts/preprocess_emb.sh "$dataset" "$EMBED_GPUS"
  done
  python -c "from preprocess.build_qwen3_embeddings import reuse_review_embeddings; reuse_review_embeddings()"
  python -u scripts/validate_yelp_amazon_pretrain.py \
    --dataset-root "$GRASPLLM_DATASET_ROOT" --stage embedded \
    --output "$GRASPLLM_DATASET_ROOT/yelp_amazon_embedding_audit.json"
}

run_stage1() {
  for dataset in "${ALL_DATASETS[@]}"; do
    require_file "$GRASPLLM_DATASET_ROOT/$dataset/processed_data.pt"
    require_file "$GRASPLLM_DATASET_ROOT/$dataset/qwen3_emb_x.pt"
  done
  GPU="$STAGE1_GPU" DATASETS="${ALL_DATASETS[*]}" STEPS_PER_DATASET=1 \
    NUM_EPOCHS=${NUM_EPOCHS:-300} LR=${LR:-1e-4} NUM_SAMPLES=${NUM_SAMPLES:-2000} SEED=${SEED:-42} \
    MODEL_SAVE_PATH="$GRASPLLM_CHECKPOINT_ROOT/structure_learner_yelp_amazon_qwen3.pth" \
    bash scripts/stage1_gnn_pretrain.sh
}

run_stage2() {
  export GRASPLLM_CHECKPOINT_ROOT
  local checkpoint="$GRASPLLM_CHECKPOINT_ROOT/structure_learner_yelp_amazon_qwen3.pth"
  require_file "$checkpoint"
  export GRASPLLM_STAGE1_CHECKPOINT="$checkpoint"
  for dataset in "${ALL_DATASETS[@]}"; do
    local train_samples=10000
    local extra=()
    [[ "$dataset" == yelpzip_* ]] && train_samples=60000 && extra+=(--large-graph)
    GPU="$STAGE2_GPUS" bash scripts/stage2_generate_seqs.sh "$dataset" \
      "${extra[@]}" \
      --motif-parallel-gpus "$STAGE2_LOGICAL_GPUS" \
      --gnn-chunk-size "${GNN_CHUNK_SIZE:-8192}" \
      --train-samples "$train_samples" --sampling-seed 42
  done
  python -u scripts/validate_yelp_amazon_pretrain.py \
    --dataset-root "$GRASPLLM_DATASET_ROOT" --stage ocs \
    --output "$GRASPLLM_DATASET_ROOT/yelp_amazon_ocs_leakage_audit.json"
}

run_stage3() {
  [[ -d "$BASE_MODEL" ]] || { echo "missing base model: $BASE_MODEL" >&2; exit 2; }
  for dataset in "${ALL_DATASETS[@]}"; do
    require_file "$GRASPLLM_DATASET_ROOT/$dataset/ocs_train.jsonl"
  done
  RUN_NAME="$PROJECTOR_NAME" bash scripts/stage3_train.sh \
    --backbone qwen3 --source "$SOURCE_SPEC" --base-model "$BASE_MODEL" \
    --gpus "$STAGE3_GPUS" --batch-size 1 --grad-accum 2 \
    --lr 5e-4 --epochs 1 --max-len 2048 --precision fp16
}

run_support() {
  for relation in yelpzip_rur yelpzip_rbr; do
    python -m experiments.yelpzip_fewshot.prepare \
      --processed-data "$GRASPLLM_DATASET_ROOT/$relation/processed_data.pt" \
      --ocs-validation "$GRASPLLM_DATASET_ROOT/$relation/ocs_val.jsonl" \
      --output-dir "$SUPPORT_ROOT/$relation" \
      --shots 1 5 10 --seeds 42 43 44 45 46
  done
}

run_eval() {
  require_file "$PROJECTOR_PATH/mm_projector.bin"
  NO_ICL_DIR_NAME=no_icl bash scripts/run_yelp_all_evals.sh \
    --model-path "$PROJECTOR_PATH" --model-base "$BASE_MODEL" \
    --dataset-root "$GRASPLLM_DATASET_ROOT" --support-root "$SUPPORT_ROOT" \
    --output-dir "$RESULT_ROOT" --gpus "${EVAL_GPUS:-0,1}"
}

case "$STAGE" in
  prepare) run_prepare ;;
  embed) run_embed ;;
  stage1) run_stage1 ;;
  stage2) run_stage2 ;;
  stage3) run_stage3 ;;
  support) run_support ;;
  eval) run_eval ;;
  all)
    run_prepare
    run_embed
    run_stage1
    run_stage2
    run_stage3
    run_support
    run_eval
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
