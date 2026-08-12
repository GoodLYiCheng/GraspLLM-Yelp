set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

RAW_PATH=${RAW_PATH:?'set RAW_PATH to yelpzip.csv'}
ARTIFACT_DIR=${ARTIFACT_DIR:-"$REPO/artifacts/grasp_dual_relation/yelpzip"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"$REPO/checkpoints/grasp_dual_relation/yelpzip"}
METHOD=${METHOD:-ocs}

python -u -m experiments.grasp_dual_relation.generate_contexts \
  --raw-path "$RAW_PATH" \
  --graph-bundle "$ARTIFACT_DIR/temporal_graphs.npz" \
  --text-embedding "$ARTIFACT_DIR/qwen3_emb_x.pt" \
  --user-structure-embedding "$CHECKPOINT_DIR/user_structure_emb.pt" \
  --business-structure-embedding "$CHECKPOINT_DIR/business_structure_emb.pt" \
  --output-dir "$ARTIFACT_DIR/contexts" \
  --method "$METHOD" \
  --max-depth 1 --user-k 8 --business-k 8 \
  --beta-user 0.55 --beta-business 0.55 --seed "${SEED:-42}"

