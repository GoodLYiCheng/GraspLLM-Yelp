set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

GRAPH_BUNDLE=${GRAPH_BUNDLE:-"$REPO/artifacts/grasp_dual_relation/yelpzip/temporal_graphs.npz"}
TEXT_EMBEDDING=${TEXT_EMBEDDING:-"$REPO/artifacts/grasp_dual_relation/yelpzip/qwen3_emb_x.pt"}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO/checkpoints/grasp_dual_relation/yelpzip"}
GPU=${GPU:-0}

CUDA_VISIBLE_DEVICES="$GPU" python -u -m experiments.grasp_dual_relation.pretrain_gnn \
  --graph-bundle "$GRAPH_BUNDLE" \
  --embedding "$TEXT_EMBEDDING" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "${EPOCHS:-300}" \
  --max-nodes "${MAX_NODES:-4000}" \
  --lr "${LR:-1e-4}" \
  --seed "${SEED:-42}" \
  --device cuda

CUDA_VISIBLE_DEVICES="$GPU" python -u -m experiments.grasp_dual_relation.infer_gnn \
  --graph-bundle "$GRAPH_BUNDLE" \
  --embedding "$TEXT_EMBEDDING" \
  --checkpoint-dir "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "${INFER_BATCH_SIZE:-256}" \
  --device cuda

