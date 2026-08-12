#!/usr/bin/env bash
set -euo pipefail

# Fine-tune only the original single-graph Projector on a balanced YelpZip support set.
# The LLM and Motif-GNN remain frozen; initialization is the Arxiv zero-shot projector.
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

BASE_MODEL=${BASE_MODEL:-/data/Qwen/Qwen3-8B}
SUPPORT_PATH=${SUPPORT_PATH:?SUPPORT_PATH must point to support_train.jsonl}
GRAPH_EMBEDDING=${GRAPH_EMBEDDING:?GRAPH_EMBEDDING must point to qwen3_emb_x.pt}
PRETRAINED_PROJECTOR=${PRETRAINED_PROJECTOR:?PRETRAINED_PROJECTOR must point to Arxiv mm_projector.bin}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
GPU=${GPU:-0}

for file in "$SUPPORT_PATH" "$GRAPH_EMBEDDING" "$PRETRAINED_PROJECTOR"; do
  [[ -f "$file" ]] || { echo "required file not found: $file" >&2; exit 2; }
done
[[ -d "$BASE_MODEL" ]] || { echo "base model directory not found: $BASE_MODEL" >&2; exit 2; }
[[ "$(basename "$OUTPUT_DIR")" == *grasp* ]] || {
  echo "OUTPUT_DIR basename must contain 'grasp' so the original evaluator loads it as GraspLLM" >&2
  exit 2
}

CUDA_VISIBLE_DEVICES="$GPU" python -u train/train_mem.py \
  --model_name_or_path "$BASE_MODEL" \
  --version v1 \
  --data_path "$SUPPORT_PATH" \
  --graph_embedding_path "$GRAPH_EMBEDDING" \
  --pretrained_projector_path "$PRETRAINED_PROJECTOR" \
  --mm_hidden_size 4096 --mm_projector_type vicuna_2layermh \
  --dual_graph_projector False \
  --tune_mm_mlp_adapter True \
  --mm_use_graph_start_end False --mm_use_graph_patch_token False \
  --bf16 True --tf32 True --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "${EPOCHS:-20}" --max_steps "${MAX_STEPS:-200}" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --save_strategy epoch --save_total_limit 1 --learning_rate "${LR:-5e-5}" \
  --warmup_ratio 0.0 --lr_scheduler_type cosine --logging_steps 1 \
  --model_max_length 4096 --gradient_checkpointing True \
  --lazy_preprocess True --report_to none
