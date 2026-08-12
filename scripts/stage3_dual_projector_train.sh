set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

BASE_MODEL=${BASE_MODEL:-${GRASP_DUAL_BASE_MODEL:-/data/Qwen/Qwen3-8B}}
DATA_PATH=${DATA_PATH:-"$REPO/artifacts/grasp_dual_relation/yelpzip/contexts/ocs_alignment.jsonl"}
GRAPH_EMBEDDING=${GRAPH_EMBEDDING:-"$REPO/artifacts/grasp_dual_relation/yelpzip/qwen3_emb_x.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO/checkpoints/grasp_dual_relation/yelpzip/projector"}
GPU=${GPU:-0}
DUAL_GRAPH_PROJECTOR=${DUAL_GRAPH_PROJECTOR:-True}

if [[ ! -d "$BASE_MODEL" ]]; then
  echo "base model directory not found: $BASE_MODEL" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="$GPU" python -u train/train_mem.py \
  --model_name_or_path "$BASE_MODEL" \
  --version v1 \
  --data_path "$DATA_PATH" \
  --graph_embedding_path "$GRAPH_EMBEDDING" \
  --mm_hidden_size 4096 \
  --mm_projector_type vicuna_2layermh \
  --dual_graph_projector "$DUAL_GRAPH_PROJECTOR" \
  --relation_graph_tokens 4 \
  --tune_mm_mlp_adapter True \
  --mm_use_graph_start_end False \
  --mm_use_graph_patch_token False \
  --bf16 True --tf32 True \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "${EPOCHS:-1}" \
  --max_steps "${MAX_STEPS:--1}" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-16}" \
  --save_strategy epoch --save_total_limit 1 \
  --learning_rate "${LR:-5e-4}" \
  --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --logging_steps 10 --model_max_length 4096 \
  --gradient_checkpointing True --lazy_preprocess True --report_to none
