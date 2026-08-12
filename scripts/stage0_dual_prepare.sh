set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

RAW_PATH=${RAW_PATH:?'set RAW_PATH to yelpzip.csv'}
EMBED_MODEL=${EMBED_MODEL:-${GRASP_DUAL_EMBED_MODEL:-/data/Qwen3-Embedding-8B}}
ARTIFACT_DIR=${ARTIFACT_DIR:-"$REPO/artifacts/grasp_dual_relation/yelpzip"}
GPU=${GPU:-0}

if [[ ! -d "$EMBED_MODEL" ]]; then
  echo "embedding model directory not found: $EMBED_MODEL" >&2
  exit 2
fi

python -u -m experiments.grasp_dual_relation.cli \
  --raw-path "$RAW_PATH" --output-dir "$ARTIFACT_DIR"

CUDA_VISIBLE_DEVICES="$GPU" python -u -m experiments.grasp_dual_relation.encode_text \
  --raw-path "$RAW_PATH" \
  --model-path "$EMBED_MODEL" \
  --output "$ARTIFACT_DIR/qwen3_emb_x.pt" \
  --max-length "${MAX_LENGTH:-512}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --device cuda
