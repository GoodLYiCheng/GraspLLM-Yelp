#!/usr/bin/env bash
set -euo pipefail

# Required paths. The runner refuses to infer provenance-sensitive assets.
: "${FIGRAPH_RAW_ROOT:?set FIGRAPH_RAW_ROOT}"
: "${FIGRAPH_OUTPUT_ROOT:?set FIGRAPH_OUTPUT_ROOT}"
: "${QWEN3_LLM:?set QWEN3_LLM to Qwen3-8B}"
: "${QWEN3_EMBED:?set QWEN3_EMBED to Qwen3-Embedding-8B}"
: "${GNN_CHECKPOINT:?set GNN_CHECKPOINT to a non-FiGraph Stage-1 checkpoint}"
: "${PROJECTOR_PATH:?set PROJECTOR_PATH to the frozen GraspLLM projector directory}"
: "${PROJECTOR_PROVENANCE:?set PROJECTOR_PROVENANCE to the explicit source manifest}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
MODE="${MODE:-smoke}"                    # smoke | pilot | full
NO_PROGRESS="${NO_PROGRESS:-0}"
PROGRESS_ARG=()
[[ "$NO_PROGRESS" == "1" ]] && PROGRESS_ARG+=(--no-progress)

DATA_DIR="$FIGRAPH_OUTPUT_ROOT/data"
SUPPORT_DIR="$FIGRAPH_OUTPUT_ROOT/support"
CONTEXT_ROOT="$FIGRAPH_OUTPUT_ROOT/contexts"
PRED_ROOT="$FIGRAPH_OUTPUT_ROOT/predictions/$MODE"
REPORT_ROOT="$FIGRAPH_OUTPUT_ROOT/reports/$MODE"
mkdir -p "$DATA_DIR" "$SUPPORT_DIR" "$CONTEXT_ROOT" "$PRED_ROOT" "$REPORT_ROOT"

if [[ "$MODE" == "full" ]]; then
  GATE_JSON="$FIGRAPH_OUTPUT_ROOT/reports/pilot/gate_k5.json"
  [[ -f "$GATE_JSON" ]] || { echo "missing pilot Gate: $GATE_JSON" >&2; exit 3; }
  "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p.get("status")=="PASS" else 4)' "$GATE_JSON"
fi

# Preflight is idempotent and preserves the graph-free text-only contract.
"$PYTHON_BIN" -m experiments.figraph_graspllm.prepare \
  --raw-root "$FIGRAPH_RAW_ROOT" --output-dir "$DATA_DIR" --motif-mode grasp_legacy
"$PYTHON_BIN" -m experiments.figraph_graspllm.audit_tokens \
  --processed-data "$DATA_DIR/processed_data.pt" --model-path "$QWEN3_EMBED" \
  --output "$DATA_DIR/qwen3_token_audit.json"
"$PYTHON_BIN" -m experiments.figraph_graspllm.encode_text \
  --processed-data "$DATA_DIR/processed_data.pt" --model-path "$QWEN3_EMBED" \
  --output "$DATA_DIR/qwen3_embedding.pt" --max-length 32768 --device "$DEVICE" "${PROGRESS_ARG[@]}"
"$PYTHON_BIN" -m experiments.figraph_graspllm.support \
  --data "$DATA_DIR/processed_data.pt" --output-dir "$SUPPORT_DIR" \
  --k 1 5 10 --seeds $(seq 42 61)

case "$MODE" in
  smoke) KS=(0 1 5 10); SEEDS=(42); VAL_ROWS=100; TEST_ROWS=100 ;;
  pilot) KS=(0 5); SEEDS=(42 43 44 45 46); VAL_ROWS=1000; TEST_ROWS=2000 ;;
  full) KS=(1 5 10); SEEDS=($(seq 42 61)); VAL_ROWS=""; TEST_ROWS="" ;;
  *) echo "MODE must be smoke, pilot, or full" >&2; exit 2 ;;
esac

run_method () {
  local method="$1" k="$2" seed="$3" split="$4" max_rows="$5" contexts="$6"
  local output="$PRED_ROOT/${method}_k${k}_seed${seed}_${split}.jsonl"
  local args=(--method "$method" --text-cohort "$DATA_DIR/text_cohort.pt"
    --model-base "$QWEN3_LLM" --output "$output" --k "$k" --seed "$seed"
    --split "$split" --max-length 32768 --device "$DEVICE" "${PROGRESS_ARG[@]}")
  [[ -n "$max_rows" ]] && args+=(--max-rows "$max_rows")
  [[ "$k" != "0" ]] && args+=(--support-manifest "$SUPPORT_DIR/support_k${k}_seed${seed}.json")
  if [[ "$method" == "full_graspllm" || "$method" == "random_graph_llm" ]]; then
    args+=(--model-path "$PROJECTOR_PATH" --projector-provenance "$PROJECTOR_PROVENANCE"
      --contexts "$contexts" --graph-embedding "$CONTEXT_ROOT/seed${seed}/structure_embedding.pt")
  fi
  "$PYTHON_BIN" -m experiments.figraph_graspllm.evaluate_llm "${args[@]}"
}

for seed in "${SEEDS[@]}"; do
  context_dir="$CONTEXT_ROOT/seed${seed}"
  "$PYTHON_BIN" -m experiments.figraph_graspllm.contexts \
    --data "$DATA_DIR/processed_data.pt" --text-embedding "$DATA_DIR/qwen3_embedding.pt" \
    --gnn-checkpoint "$GNN_CHECKPOINT" --output-dir "$context_dir" --device "$DEVICE" --seed "$seed"
  for k in "${KS[@]}"; do
    for split in validation test; do
      rows="$VAL_ROWS"; [[ "$split" == "test" ]] && rows="$TEST_ROWS"
      run_method text_only_matched "$k" "$seed" "$split" "$rows" ""
      run_method text_only_maxcontext "$k" "$seed" "$split" "$rows" ""
      run_method random_graph_llm "$k" "$seed" "$split" "$rows" "$context_dir/contexts_random_seed${seed}.jsonl"
      run_method full_graspllm "$k" "$seed" "$split" "$rows" "$context_dir/contexts_ocs_seed${seed}.jsonl"
    done
    if [[ "$k" != "0" ]]; then
      lr_dir="$PRED_ROOT/frozen_lr_k${k}_seed${seed}"
      "$PYTHON_BIN" -m experiments.figraph_graspllm.frozen_lr \
        --text-cohort "$DATA_DIR/text_cohort.pt" \
        --structure-embedding "$context_dir/structure_embedding.pt" \
        --gnn-checkpoint "$GNN_CHECKPOINT" \
        --support-manifest "$SUPPORT_DIR/support_k${k}_seed${seed}.json" \
        --output-dir "$lr_dir" --seed "$seed"
      "$PYTHON_BIN" -m experiments.figraph_graspllm.score_results \
        --validation "$lr_dir/frozen_motifgnn_lr_validation.jsonl" \
        --test "$lr_dir/frozen_motifgnn_lr_test.jsonl" \
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
