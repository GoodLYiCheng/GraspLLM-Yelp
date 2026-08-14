#!/usr/bin/env bash
set -euo pipefail

: "${FIGRAPH_RAW_ROOT:?set FIGRAPH_RAW_ROOT}"
: "${FIGRAPH_OUTPUT_ROOT:?set FIGRAPH_OUTPUT_ROOT}"
: "${QWEN3_LLM:?set QWEN3_LLM to Qwen3-8B}"
: "${QWEN3_EMBED:?set QWEN3_EMBED to Qwen3-Embedding-8B}"
: "${GNN_CHECKPOINT:?set GNN_CHECKPOINT to a non-FiGraph Stage-1 checkpoint}"
: "${PROJECTOR_PATH:?set PROJECTOR_PATH to the frozen GraspLLM projector directory}"
: "${PROJECTOR_PROVENANCE:?set PROJECTOR_PROVENANCE to the explicit source manifest}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${MODE:-smoke}"                     # smoke | pilot | full
V100_GPU_IDS="${V100_GPU_IDS:-0,1,2,3}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
NO_PROGRESS="${NO_PROGRESS:-1}"
FORCE_PREFLIGHT="${FORCE_PREFLIGHT:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
IFS=',' read -r -a GPU_IDS <<< "$V100_GPU_IDS"
[[ "${#GPU_IDS[@]}" -eq 4 ]] || { echo "V100_GPU_IDS must contain exactly four GPU IDs" >&2; exit 2; }
[[ "$MAX_LENGTH" == "16384" ]] || {
  echo "the audited V100 profile fixes MAX_LENGTH=16384; use run_figraph_mvp.sh for the native-32K profile" >&2; exit 2;
}

PROFILE="v100x4_fp16_${MAX_LENGTH}"
DATA_DIR="$FIGRAPH_OUTPUT_ROOT/data"
SUPPORT_DIR="$FIGRAPH_OUTPUT_ROOT/support"
CONTEXT_ROOT="$FIGRAPH_OUTPUT_ROOT/contexts/$PROFILE"
PRED_ROOT="$FIGRAPH_OUTPUT_ROOT/predictions/$PROFILE/$MODE"
REPORT_ROOT="$FIGRAPH_OUTPUT_ROOT/reports/$PROFILE/$MODE"
LOG_ROOT="$FIGRAPH_OUTPUT_ROOT/logs/$PROFILE/$MODE"
EMBED_DIR="$DATA_DIR/embedding_shards/$PROFILE"
EMBEDDING="$DATA_DIR/qwen3_embedding_${PROFILE}.pt"
mkdir -p "$DATA_DIR" "$SUPPORT_DIR" "$CONTEXT_ROOT" "$PRED_ROOT" "$REPORT_ROOT" "$LOG_ROOT" "$EMBED_DIR"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader > "$REPORT_ROOT/hardware.csv"
for gpu in "${GPU_IDS[@]}"; do
  name="$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader | head -n 1)"
  memory_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  [[ "$name" == *V100* ]] || { echo "GPU $gpu is not a V100: $name" >&2; exit 2; }
  [[ "$memory_mib" -ge 30000 ]] || {
    echo "GPU $gpu has only ${memory_mib} MiB; this FP16 Qwen3-8B profile requires 32GB V100 cards" >&2; exit 2;
  }
done

if [[ "$MODE" == "full" ]]; then
  GATE_JSON="$FIGRAPH_OUTPUT_ROOT/reports/$PROFILE/pilot/gate_k5.json"
  [[ -f "$GATE_JSON" ]] || { echo "missing pilot Gate: $GATE_JSON" >&2; exit 3; }
  "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p.get("status")=="PASS" else 4)' "$GATE_JSON"
fi

if [[ "$FORCE_PREFLIGHT" == "1" || ! -f "$DATA_DIR/processed_data.pt" || ! -f "$DATA_DIR/text_cohort.pt" ]]; then
  "$PYTHON_BIN" -m experiments.figraph_graspllm.prepare \
    --raw-root "$FIGRAPH_RAW_ROOT" --output-dir "$DATA_DIR" --motif-mode grasp_legacy
fi
if [[ "$FORCE_PREFLIGHT" == "1" || ! -f "$DATA_DIR/qwen3_token_audit.json" ]]; then
  "$PYTHON_BIN" -m experiments.figraph_graspllm.audit_tokens \
    --processed-data "$DATA_DIR/processed_data.pt" --model-path "$QWEN3_EMBED" \
    --output "$DATA_DIR/qwen3_token_audit.json"
fi

reuse_embedding=0
if [[ "$FORCE_PREFLIGHT" != "1" && -f "$EMBEDDING" ]]; then
  if "$PYTHON_BIN" -c 'import sys,torch; from experiments.figraph_graspllm.artifacts import file_sha256; o=torch.load(sys.argv[1],map_location="cpu",weights_only=False); m=o["metadata"]; ok=(int(m.get("final_max_length",-1))==int(sys.argv[3]) and int(m.get("merged_from_shards",-1))==4 and m.get("processed_data_sha256")==file_sha256(sys.argv[2])); raise SystemExit(0 if ok else 1)' "$EMBEDDING" "$DATA_DIR/processed_data.pt" "$MAX_LENGTH"; then
    reuse_embedding=1
  fi
fi
if [[ "$reuse_embedding" -eq 0 ]]; then
  pids=()
  for shard_id in 0 1 2 3; do
    gpu="${GPU_IDS[$shard_id]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m experiments.figraph_graspllm.encode_text \
      --processed-data "$DATA_DIR/processed_data.pt" --model-path "$QWEN3_EMBED" \
      --output "$EMBED_DIR/shard${shard_id}.pt" --max-length "$MAX_LENGTH" --device cuda \
      --num-shards 4 --shard-id "$shard_id" --no-progress \
      > "$LOG_ROOT/embed_shard${shard_id}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]] || { echo "an embedding shard failed; inspect $LOG_ROOT/embed_shard*.log" >&2; exit 5; }
  "$PYTHON_BIN" -m experiments.figraph_graspllm.merge_embeddings \
    --inputs "$EMBED_DIR/shard0.pt" "$EMBED_DIR/shard1.pt" "$EMBED_DIR/shard2.pt" "$EMBED_DIR/shard3.pt" \
    --output "$EMBEDDING"
else
  echo "reusing verified embedding: $EMBEDDING"
fi
"$PYTHON_BIN" -m experiments.figraph_graspllm.support \
  --data "$DATA_DIR/processed_data.pt" --output-dir "$SUPPORT_DIR" \
  --k 1 5 10 --seeds $(seq 42 61)

case "$MODE" in
  smoke) KS=(0 1 5 10); SEEDS=(42); VAL_ROWS=100; TEST_ROWS=100 ;;
  pilot) KS=(0 5); SEEDS=(42 43 44 45 46); VAL_ROWS=1000; TEST_ROWS=2000 ;;
  full) KS=(1 5 10); SEEDS=($(seq 42 61)); VAL_ROWS=""; TEST_ROWS="" ;;
  *) echo "MODE must be smoke, pilot, or full" >&2; exit 2 ;;
esac

run_batch () {
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  pids=()
  slot=0
  [[ "$failed" -eq 0 ]]
}

pids=()
slot=0
for seed in "${SEEDS[@]}"; do
  gpu="${GPU_IDS[$slot]}"
  context_dir="$CONTEXT_ROOT/seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m experiments.figraph_graspllm.contexts \
    --data "$DATA_DIR/processed_data.pt" --text-embedding "$EMBEDDING" \
    --gnn-checkpoint "$GNN_CHECKPOINT" --output-dir "$context_dir" --device cuda --seed "$seed" \
    > "$LOG_ROOT/context_seed${seed}.log" 2>&1 &
  pids+=("$!")
  slot=$((slot + 1))
  if [[ "$slot" -eq 4 ]]; then
    run_batch || { echo "context generation failed; inspect $LOG_ROOT/context_seed*.log" >&2; exit 6; }
  fi
done
if [[ "${#pids[@]}" -gt 0 ]]; then
  run_batch || { echo "context generation failed; inspect $LOG_ROOT/context_seed*.log" >&2; exit 6; }
fi

run_method () {
  local method="$1" k="$2" seed="$3" split="$4" max_rows="$5" contexts="$6"
  local output="$PRED_ROOT/${method}_k${k}_seed${seed}_${split}.jsonl"
  local args=(--method "$method" --text-cohort "$DATA_DIR/text_cohort.pt"
    --model-base "$QWEN3_LLM" --output "$output" --k "$k" --seed "$seed"
    --split "$split" --max-length "$MAX_LENGTH" --device cuda --attn-implementation sdpa)
  [[ "$NO_PROGRESS" == "1" ]] && args+=(--no-progress)
  [[ -n "$max_rows" ]] && args+=(--max-rows "$max_rows")
  [[ "$k" != "0" ]] && args+=(--support-manifest "$SUPPORT_DIR/support_k${k}_seed${seed}.json")
  if [[ "$method" == "full_graspllm" || "$method" == "random_graph_llm" ]]; then
    args+=(--model-path "$PROJECTOR_PATH" --projector-provenance "$PROJECTOR_PROVENANCE"
      --contexts "$contexts" --graph-embedding "$CONTEXT_ROOT/seed${seed}/structure_embedding.pt")
  fi
  "$PYTHON_BIN" -m experiments.figraph_graspllm.evaluate_llm "${args[@]}"
}

pids=()
slot=0
for seed in "${SEEDS[@]}"; do
  context_dir="$CONTEXT_ROOT/seed${seed}"
  for k in "${KS[@]}"; do
    for split in validation test; do
      rows="$VAL_ROWS"; [[ "$split" == "test" ]] && rows="$TEST_ROWS"
      for method in text_only_matched text_only_maxcontext random_graph_llm full_graspllm; do
        contexts=""
        [[ "$method" == "random_graph_llm" ]] && contexts="$context_dir/contexts_random_seed${seed}.jsonl"
        [[ "$method" == "full_graspllm" ]] && contexts="$context_dir/contexts_ocs_seed${seed}.jsonl"
        gpu="${GPU_IDS[$slot]}"
        log="$LOG_ROOT/${method}_k${k}_seed${seed}_${split}.log"
        CUDA_VISIBLE_DEVICES="$gpu" run_method "$method" "$k" "$seed" "$split" "$rows" "$contexts" > "$log" 2>&1 &
        pids+=("$!")
        slot=$((slot + 1))
        if [[ "$slot" -eq 4 ]]; then
          run_batch || { echo "LLM evaluation failed; inspect $LOG_ROOT" >&2; exit 7; }
        fi
      done
    done
  done
done
if [[ "${#pids[@]}" -gt 0 ]]; then
  run_batch || { echo "LLM evaluation failed; inspect $LOG_ROOT" >&2; exit 7; }
fi

for seed in "${SEEDS[@]}"; do
  context_dir="$CONTEXT_ROOT/seed${seed}"
  for k in "${KS[@]}"; do
    if [[ "$k" != "0" ]]; then
      lr_dir="$PRED_ROOT/frozen_lr_k${k}_seed${seed}"
      "$PYTHON_BIN" -m experiments.figraph_graspllm.frozen_lr \
        --text-cohort "$DATA_DIR/text_cohort.pt" --structure-embedding "$context_dir/structure_embedding.pt" \
        --gnn-checkpoint "$GNN_CHECKPOINT" --support-manifest "$SUPPORT_DIR/support_k${k}_seed${seed}.json" \
        --output-dir "$lr_dir" --seed "$seed"
      "$PYTHON_BIN" -m experiments.figraph_graspllm.score_results \
        --validation "$lr_dir/frozen_motifgnn_lr_validation.jsonl" --test "$lr_dir/frozen_motifgnn_lr_test.jsonl" \
        --output "$REPORT_ROOT/frozen_motifgnn_lr_k${k}_seed${seed}.json"
    fi
    for method in text_only_matched text_only_maxcontext random_graph_llm full_graspllm; do
      "$PYTHON_BIN" -m experiments.figraph_graspllm.score_results \
        --validation "$PRED_ROOT/${method}_k${k}_seed${seed}_validation.jsonl" \
        --test "$PRED_ROOT/${method}_k${k}_seed${seed}_test.jsonl" \
        --output "$REPORT_ROOT/${method}_k${k}_seed${seed}.json"
    done
  done
done

if (( ${#SEEDS[@]} > 1 )); then
  for k in "${KS[@]}"; do
    for method in text_only_matched text_only_maxcontext random_graph_llm full_graspllm; do
      reports=()
      for seed in "${SEEDS[@]}"; do reports+=("$REPORT_ROOT/${method}_k${k}_seed${seed}.json"); done
      "$PYTHON_BIN" -m experiments.figraph_graspllm.summarize \
        --reports "${reports[@]}" --output "$REPORT_ROOT/${method}_k${k}_mean_std.json"
    done
    if [[ "$k" != "0" ]]; then
      reports=()
      for seed in "${SEEDS[@]}"; do reports+=("$REPORT_ROOT/frozen_motifgnn_lr_k${k}_seed${seed}.json"); done
      "$PYTHON_BIN" -m experiments.figraph_graspllm.summarize \
        --reports "${reports[@]}" --output "$REPORT_ROOT/frozen_motifgnn_lr_k${k}_mean_std.json"
    fi
  done
fi

if [[ "$MODE" == "pilot" ]]; then
  for method in text_only_matched random_graph_llm full_graspllm; do
    inputs=()
    for seed in "${SEEDS[@]}"; do inputs+=("$PRED_ROOT/${method}_k5_seed${seed}_test.jsonl"); done
    "$PYTHON_BIN" -m experiments.figraph_graspllm.aggregate \
      --inputs "${inputs[@]}" --output "$PRED_ROOT/${method}_k5_seed_mean_test.jsonl"
  done
  "$PYTHON_BIN" -m experiments.figraph_graspllm.gate \
    --full "$PRED_ROOT/full_graspllm_k5_seed_mean_test.jsonl" \
    --text-matched "$PRED_ROOT/text_only_matched_k5_seed_mean_test.jsonl" \
    --random-graph "$PRED_ROOT/random_graph_llm_k5_seed_mean_test.jsonl" \
    --output "$REPORT_ROOT/gate_k5.json" --iterations 2000
fi
