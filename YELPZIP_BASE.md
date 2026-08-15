# YelpZip original GraspLLM zero-shot base

This base uses the original Stage 0–3 pipeline with two registered static TAGs:
`yelpzip_rur` (same user) and `yelpzip_rbr` (same business). It is deliberately
transductive and does not use time. Yelp labels never train the GNN or projector.

## 1. Paths and preprocessing

```bash
export REPO=/path/to/GraspLLM
export GRASPLLM_DATASET_ROOT=$REPO/dataset
export GRASPLLM_CHECKPOINT_ROOT=$REPO/checkpoints
export QWEN3_EMB_MODEL=/data/Qwen3-Embedding-8B
export PYTHONPATH=$REPO
cd "$REPO"

python preprocess/prepare_yelpzip.py \
  --raw-path /path/to/yelpzip.csv \
  --dataset-root "$GRASPLLM_DATASET_ROOT" \
  --max-neighbors 32 --val-size 10000 --test-size 20000 --seed 42
```

## 2. Original Stage 0–3

Stage 0 encodes RUR once and automatically reuses the identical embedding for RBR.

```bash
bash scripts/preprocess_emb.sh yelpzip_rur 0

GPU=0 bash scripts/stage1_gnn_pretrain.sh

# 原始单卡路径（保持不变）
GPU=0 bash scripts/stage2_generate_seqs.sh yelpzip_rur --large-graph
GPU=0 bash scripts/stage2_generate_seqs.sh yelpzip_rbr --large-graph

# YelpZip 推荐：四卡 motif-parallel GNN 推理。GPU 编号是 CUDA_VISIBLE_DEVICES
# 过滤后的逻辑编号；--gnn-chunk-size 可调小至 4096 以进一步降低峰值显存。
GPU=0,1,2,3 bash scripts/stage2_generate_seqs.sh yelpzip_rur --large-graph \
  --motif-parallel-gpus 0,1,2,3 --gnn-chunk-size 8192
GPU=0,1,2,3 bash scripts/stage2_generate_seqs.sh yelpzip_rbr --large-graph \
  --motif-parallel-gpus 0,1,2,3 --gnn-chunk-size 8192

RUN_NAME=grasp-qwen3-qwen3emb-vicuna_2layermh-arxiv \
bash scripts/stage3_train.sh \
  --backbone qwen3 --source arxiv \
  --base-model /data/Qwen/Qwen3-8B --gpus 0
```

The Yelp directories must contain `ocs_val.jsonl` with 10,000 rows and
`ocs_test.jsonl` with 20,000 rows. They must not contain an `ocs_train.jsonl`.

## 3. Free-text Accuracy and probability metrics

```bash
export CKPT=$GRASPLLM_CHECKPOINT_ROOT/grasp-qwen3-qwen3emb-vicuna_2layermh-arxiv
export RESULT_ROOT=$REPO/artifacts/yelpzip_original_base

bash scripts/eval.sh --ckpt "$CKPT" --backbone qwen3 \
  --base-model /data/Qwen/Qwen3-8B --dataset yelpzip_rur --gpus 0
bash scripts/eval.sh --ckpt "$CKPT" --backbone qwen3 \
  --base-model /data/Qwen/Qwen3-8B --dataset yelpzip_rbr --gpus 0

python eval/eval_yelp_probability.py \
  --model-path "$CKPT" --model-base /data/Qwen/Qwen3-8B \
  --dataset yelpzip_rur --output-dir "$RESULT_ROOT/rur" --device cuda
python eval/eval_yelp_probability.py \
  --model-path "$CKPT" --model-base /data/Qwen/Qwen3-8B \
  --dataset yelpzip_rbr --output-dir "$RESULT_ROOT/rbr" --device cuda

python eval/summarize_yelp.py \
  --rur-probability "$RESULT_ROOT/rur/probability_metrics.json" \
  --rbr-probability "$RESULT_ROOT/rbr/probability_metrics.json" \
  --rur-accuracy "$CKPT/answers_yelpzip_rur.jsonl.metrics.json" \
  --rbr-accuracy "$CKPT/answers_yelpzip_rbr.jsonl.metrics.json" \
  --output "$RESULT_ROOT/summary.json"
```

## 4. Exact-mask random reference

```bash
python -m experiments.random.run_static_yelp \
  --processed-data "$GRASPLLM_DATASET_ROOT/yelpzip_rur/processed_data.pt" \
  --output-dir "$RESULT_ROOT/random"
```

Interpret PR-AUC against test fraud prevalence and ROC-AUC against 0.5. A high
F1 caused by a low validation-selected threshold is not evidence of ranking ability.

## 5. Strict 1/5/10-shot in-context evaluation

This mode reuses the preceding Arxiv-trained zero-shot checkpoint without any
Yelp parameter training.
For each relation, it samples `K` labelled reviews **per class** from the existing
validation mask (seeds 42–46), removes those reviews from threshold selection, and
inserts their text, graph context, and labels into every query prompt. The LLM,
Projector, and Stage-1 GNN are frozen. Support graphs are passed in support order,
followed by the query graph, with exactly one embedding row per `<graph>` marker.
Test labels are never read by support sampling or threshold selection.

```bash
export FS_ROOT=$REPO/artifacts/yelpzip_original_fewshot
export ZERO_SHOT_CKPT=$GRASPLLM_CHECKPOINT_ROOT/grasp-qwen3-qwen3emb-vicuna_2layermh-arxiv

for relation in yelpzip_rur yelpzip_rbr; do
  python -m experiments.yelpzip_fewshot.prepare \
    --processed-data "$GRASPLLM_DATASET_ROOT/$relation/processed_data.pt" \
    --ocs-validation "$GRASPLLM_DATASET_ROOT/$relation/ocs_val.jsonl" \
    --output-dir "$FS_ROOT/$relation"

  # Qwen3 retains complete support/query review text by default.
  # Add explicit token caps only for a constrained-context ablation.
  for k in 1 5 10; do
    for seed in 42 43 44 45 46; do
      python eval/eval_yelp_probability.py \
        --model-path "$ZERO_SHOT_CKPT" --model-base /data/Qwen/Qwen3-8B \
        --dataset "$relation" --validation-jsonl "$FS_ROOT/$relation/k${k}_seed${seed}/validation_holdout.jsonl" \
        --icl-support-jsonl "$FS_ROOT/$relation/k${k}_seed${seed}/support_train.jsonl" \
        --support-manifest "$FS_ROOT/$relation/k${k}_seed${seed}/support_manifest.json" \
        --icl-support-graphs --max-length 32768 \
        --output-dir "$FS_ROOT/$relation/results/k${k}_seed${seed}" --device cuda
    done
  done
done
```

Each result directory contains held-out validation/test predictions and
`probability_metrics.json`. Report the mean and standard deviation over seeds 42–46
for each `K`, separately from the zero-shot result.
