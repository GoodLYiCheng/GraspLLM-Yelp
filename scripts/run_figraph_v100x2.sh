#!/usr/bin/env bash
set -euo pipefail

# Two-card entry point for the shared V100 runner. Each card still loads one
# complete FP16 Qwen3-8B model, so both cards must have at least 32GB VRAM.
export V100_GPU_IDS="${V100_GPU_IDS:-0,1}"
export V100_WORKERS=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_figraph_v100x4.sh" "$@"
