# GraspLLM-Yelp Linux 端到端运行命令

本手册覆盖 YelpZip 静态基线的数据准备、Stage 0、Stage 1、Stage 2、Stage 3，以及 zero-shot 和冻结模型的 1/5/10-shot ICL 推理。RUR 与 RBR 分别运行，最终统一汇总。

> Stage 1 当前实现是单进程、单 GPU 训练，不支持 DDP。Stage 0、Stage 2、Stage 3 和最终推理支持多 GPU，具体方式见各节。

## 0. 公共环境与路径

```bash
conda activate graspllm

export REPO=/home/xiewt24/liuyc/GraspLLM-Yelp
export GRASPLLM_DATASET_ROOT=$REPO/dataset
export GRASPLLM_CHECKPOINT_ROOT=$REPO/graspllm_checkpoints
export QWEN3_EMB_MODEL=/data/Qwen3-Embedding-8B
export BASE_MODEL=/data/Qwen3-8B/Qwen3-8B
export RAW_YELP=$REPO/dataset/yelpzip.csv
export ZERO_SHOT_NAME=grasp-qwen3-arxiv-zero-shot
export ZERO_SHOT_CKPT=$GRASPLLM_CHECKPOINT_ROOT/$ZERO_SHOT_NAME
export FS_ROOT=$REPO/artifacts/yelpzip_original_fewshot
export RESULT_ROOT=$REPO/artifacts/yelpzip_complete_results/few-shot
export PYTHONPATH=$REPO
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$GRASPLLM_CHECKPOINT_ROOT" "$FS_ROOT" "$RESULT_ROOT/logs"
```

模型目录必须直接包含 `config.json`。先检查路径：

```bash
test -f "$QWEN3_EMB_MODEL/config.json" && echo '[OK] embedding model' || echo '[MISSING] embedding model'
test -f "$BASE_MODEL/config.json" && echo '[OK] base model' || echo '[MISSING] base model'
test -f "$RAW_YELP" && echo '[OK] YelpZip CSV' || echo '[MISSING] YelpZip CSV'
nvidia-smi
```

如果模型路径不确定：

```bash
find /data -name config.json -path '*Qwen3*' -print 2>/dev/null
```

## 1. YelpZip 静态双关系数据准备（CPU）

该步骤生成共享节点顺序和 mask 的 `yelpzip_rur`、`yelpzip_rbr`，不使用时间信息。默认验证集 10,000 条、测试集 20,000 条。

```bash
cd "$REPO"

python preprocess/prepare_yelpzip.py \
  --raw-path "$RAW_YELP" \
  --dataset-root "$GRASPLLM_DATASET_ROOT" \
  --max-neighbors 32 \
  --val-size 10000 \
  --test-size 20000 \
  --seed 42
```

如果目标目录已经存在且确实要重新生成，才追加 `--overwrite`。不要在已有正式数据上随意使用该参数。

检查：

```bash
ls -lh "$GRASPLLM_DATASET_ROOT/yelpzip_rur/processed_data.pt"
ls -lh "$GRASPLLM_DATASET_ROOT/yelpzip_rbr/processed_data.pt"
```

## 2. Stage 0：Qwen3-Embedding-8B 文本编码

YelpZip 的 RUR/RBR 文本与节点顺序一致，因此只编码 RUR，随后通过元数据哈希校验复用到 RBR。

### 2.1 单卡

V100 32 GB 建议从 `BATCH_SIZE=8` 开始；A100 40 GB 可从 `BATCH_SIZE=16` 或 `32` 开始。如果 OOM，继续减半。

```bash
cd "$REPO"

QWEN3_ATTN_IMPLEMENTATION=eager BATCH_SIZE=8 \
  bash scripts/preprocess_emb.sh yelpzip_rur 0
```

在 A100 上可使用自动 attention 后端：

```bash
QWEN3_ATTN_IMPLEMENTATION=auto BATCH_SIZE=32 \
  bash scripts/preprocess_emb.sh yelpzip_rur 0
```

### 2.2 多卡分片编码

四卡 V100 32 GB：

```bash
cd "$REPO"

QWEN3_ATTN_IMPLEMENTATION=eager BATCH_SIZE=8 \
  bash scripts/preprocess_emb.sh yelpzip_rur 0,1,2,3

python -c "from preprocess.build_qwen3_embeddings import reuse_yelpzip_embedding; reuse_yelpzip_embedding()"
```

四卡 A100 40 GB：

```bash
QWEN3_ATTN_IMPLEMENTATION=auto BATCH_SIZE=32 \
  bash scripts/preprocess_emb.sh yelpzip_rur 0,1,2,3

python -c "from preprocess.build_qwen3_embeddings import reuse_yelpzip_embedding; reuse_yelpzip_embedding()"
```

多卡模式按节点分片，每张卡独立编码，最后自动合并 `qwen3_emb_x.pt`。显式复用命令用于确保多卡合并后 RBR 也获得相同 embedding。

检查：

```bash
ls -lh "$GRASPLLM_DATASET_ROOT/yelpzip_rur/qwen3_emb_x.pt"
ls -lh "$GRASPLLM_DATASET_ROOT/yelpzip_rbr/qwen3_emb_x.pt"
sha256sum \
  "$GRASPLLM_DATASET_ROOT/yelpzip_rur/qwen3_emb_x.pt" \
  "$GRASPLLM_DATASET_ROOT/yelpzip_rbr/qwen3_emb_x.pt"
```

## 3. Stage 1：Motif-GNN 自监督预训练

Stage 1 不使用 Yelp 标签，也不在 Yelp 上训练。默认源数据集为：

```text
arxiv pubmed computer history reddit
```

运行前，这五个数据集均需存在 `processed_data.pt` 和 `qwen3_emb_x.pt`：

```bash
for dataset in ogbn-arxiv pubmed computer history reddit; do
  test -f "$GRASPLLM_DATASET_ROOT/$dataset/processed_data.pt" || echo "[MISSING] $dataset/processed_data.pt"
  test -f "$GRASPLLM_DATASET_ROOT/$dataset/qwen3_emb_x.pt" || echo "[MISSING] $dataset/qwen3_emb_x.pt"
done
```

如果源数据集还没有 embedding，可分别执行 Stage 0，例如：

```bash
bash scripts/preprocess_emb.sh arxiv 0,1,2,3
bash scripts/preprocess_emb.sh pubmed 0,1,2,3
bash scripts/preprocess_emb.sh computer 0,1,2,3
bash scripts/preprocess_emb.sh history 0,1,2,3
bash scripts/preprocess_emb.sh reddit 0,1,2,3
```

### 3.1 当前正式命令：单卡

```bash
cd "$REPO"

GPU=0 \
DATASETS="arxiv pubmed computer history reddit" \
NUM_EPOCHS=300 \
LR=1e-4 \
NUM_SAMPLES=2000 \
SEED=0 \
bash scripts/stage1_gnn_pretrain.sh
```

输出：

```text
$GRASPLLM_CHECKPOINT_ROOT/structure_learner_qwen3.pth
```

### 3.2 多卡说明

当前 `scripts/stage1_gnn_pretrain.sh` 和 `gnn/train.py` 没有实现 DDP/DataParallel。因此没有与单卡训练等价的多卡命令，下面这种写法是错误的：

```bash
# 不要执行：Stage 1 不会因此使用四张卡
GPU=0,1,2,3 bash scripts/stage1_gnn_pretrain.sh
```

请使用上一节的单卡命令。多张空闲 GPU 可留给 Stage 0、Stage 2、Stage 3 或其他独立实验。

## 4. Stage 2：RUR/RBR 图表示与 OCS 序列生成

前置文件：

```bash
test -f "$GRASPLLM_CHECKPOINT_ROOT/structure_learner_qwen3.pth" || echo '[MISSING] Stage-1 checkpoint'
test -f "$GRASPLLM_DATASET_ROOT/yelpzip_rur/qwen3_emb_x.pt" || echo '[MISSING] RUR embedding'
test -f "$GRASPLLM_DATASET_ROOT/yelpzip_rbr/qwen3_emb_x.pt" || echo '[MISSING] RBR embedding'
```

### 4.1 单卡顺序执行

```bash
cd "$REPO"

GPU=0 bash scripts/stage2_generate_seqs.sh yelpzip_rur \
  --large-graph --gnn-chunk-size 4096

GPU=0 bash scripts/stage2_generate_seqs.sh yelpzip_rbr \
  --large-graph --gnn-chunk-size 4096
```

### 4.2 四卡 motif-parallel：每个关系依次使用四卡

```bash
cd "$REPO"

GPU=0,1,2,3 bash scripts/stage2_generate_seqs.sh yelpzip_rur \
  --large-graph \
  --motif-parallel-gpus 0,1,2,3 \
  --gnn-chunk-size 8192

GPU=0,1,2,3 bash scripts/stage2_generate_seqs.sh yelpzip_rbr \
  --large-graph \
  --motif-parallel-gpus 0,1,2,3 \
  --gnn-chunk-size 8192
```

若 V100 显存仍不足，把 `--gnn-chunk-size 8192` 改为 `4096` 或 `2048`。

### 4.3 四卡并发：RUR 使用 0/1，RBR 使用 2/3

```bash
cd "$REPO"
mkdir -p "$RESULT_ROOT/logs"

GPU=0,1 bash scripts/stage2_generate_seqs.sh yelpzip_rur \
  --large-graph --motif-parallel-gpus 0,1 --gnn-chunk-size 4096 \
  > "$RESULT_ROOT/logs/stage2_rur.log" 2>&1 &
rur_pid=$!

GPU=2,3 bash scripts/stage2_generate_seqs.sh yelpzip_rbr \
  --large-graph --motif-parallel-gpus 0,1 --gnn-chunk-size 4096 \
  > "$RESULT_ROOT/logs/stage2_rbr.log" 2>&1 &
rbr_pid=$!

wait "$rur_pid"
wait "$rbr_pid"
```

第二个进程设置 `GPU=2,3` 后只看见两张卡，所以 `--motif-parallel-gpus` 仍使用逻辑编号 `0,1`。

检查输出数量：

```bash
for relation in yelpzip_rur yelpzip_rbr; do
  wc -l \
    "$GRASPLLM_DATASET_ROOT/$relation/ocs_val.jsonl" \
    "$GRASPLLM_DATASET_ROOT/$relation/ocs_test.jsonl"
done
```

每个关系应分别得到 10,000 条 validation 和 20,000 条 test；Yelp 不生成 `ocs_train.jsonl`。

## 5. Stage 3：在 Arxiv 上训练 Projector

Stage 3 只读取 Arxiv 的 `ocs_train.jsonl`，不会用 Yelp 标签训练。LLM 主干冻结，仅训练 Projector。

前置检查：

```bash
test -f "$GRASPLLM_DATASET_ROOT/ogbn-arxiv/ocs_train.jsonl" || echo '[MISSING] Arxiv ocs_train.jsonl'
test -f "$BASE_MODEL/config.json" || echo '[MISSING] Qwen3-8B config.json'
```

如果 Arxiv 尚未生成 OCS：

```bash
GPU=0,1,2,3 bash scripts/stage2_generate_seqs.sh arxiv \
  --motif-parallel-gpus 0,1,2,3 \
  --gnn-chunk-size 8192
```

### 5.1 单卡 V100 32 GB 安全配置

```bash
cd "$REPO"

RUN_NAME=$ZERO_SHOT_NAME \
bash scripts/stage3_train.sh \
  --backbone qwen3 \
  --source arxiv \
  --base-model "$BASE_MODEL" \
  --gpus 0 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-4 \
  --epochs 1 \
  --max-len 2048 \
  --precision fp16
```

如果仍然 OOM，先把 `--max-len 2048` 降到 `1024`；不要增大 batch size。

### 5.2 单卡 A100 40 GB

```bash
RUN_NAME=$ZERO_SHOT_NAME \
bash scripts/stage3_train.sh \
  --backbone qwen3 \
  --source arxiv \
  --base-model "$BASE_MODEL" \
  --gpus 0 \
  --batch-size 2 \
  --grad-accum 4 \
  --lr 5e-4 \
  --epochs 1 \
  --max-len 4096 \
  --precision bf16
```

### 5.3 四卡 V100 32 GB（DDP）

```bash
cd "$REPO"

RUN_NAME=$ZERO_SHOT_NAME \
bash scripts/stage3_train.sh \
  --backbone qwen3 \
  --source arxiv \
  --base-model "$BASE_MODEL" \
  --gpus 0,1,2,3 \
  --batch-size 1 \
  --grad-accum 2 \
  --lr 5e-4 \
  --epochs 1 \
  --max-len 2048 \
  --precision fp16
```

### 5.4 四卡 A100 40 GB（DDP）

```bash
RUN_NAME=$ZERO_SHOT_NAME \
bash scripts/stage3_train.sh \
  --backbone qwen3 \
  --source arxiv \
  --base-model "$BASE_MODEL" \
  --gpus 0,1,2,3 \
  --batch-size 2 \
  --grad-accum 1 \
  --lr 5e-4 \
  --epochs 1 \
  --max-len 4096 \
  --precision bf16
```

检查 Projector：

```bash
ls -lh "$ZERO_SHOT_CKPT/mm_projector.bin"
```

## 6. 生成 1/5/10-shot support（CPU，仅首次执行）

如果已经生成 support，可以跳过本节。

```bash
cd "$REPO"

for relation in yelpzip_rur yelpzip_rbr; do
  python -m experiments.yelpzip_fewshot.prepare \
    --processed-data "$GRASPLLM_DATASET_ROOT/$relation/processed_data.pt" \
    --ocs-validation "$GRASPLLM_DATASET_ROOT/$relation/ocs_val.jsonl" \
    --output-dir "$FS_ROOT/$relation" \
    --shots 1 5 10 \
    --seeds 42 43 44 45 46
done
```

## 7. Zero-shot 与 1/5/10-shot ICL 推理

该阶段使用 GPU。RUR 和 RBR 可分别放在 GPU 0、GPU 1 上并行运行。ICL 不重新训练模型，只把带标签的 support 文本及其 `<graph>` embedding 放入 prompt。

### 7.1 完整推理：zero-shot + 1/5/10-shot

```bash
cd "$REPO"
mkdir -p "$RESULT_ROOT/logs"

nohup bash scripts/run_yelp_all_evals.sh \
  --model-path "$ZERO_SHOT_CKPT" \
  --model-base "$BASE_MODEL" \
  --dataset-root "$GRASPLLM_DATASET_ROOT" \
  --support-root "$FS_ROOT" \
  --output-dir "$RESULT_ROOT" \
  --gpus 0,1 \
  --max-validation-queries 1000 \
  > "$RESULT_ROOT/logs/zero_and_icl_all.log" 2>&1 &

echo $! | tee "$RESULT_ROOT/logs/zero_and_icl_all.pid"
tail -f "$RESULT_ROOT/logs/zero_and_icl_all.log"
```

### 7.2 Zero-shot 已完成：只运行 1/5/10-shot ICL

```bash
cd "$REPO"
mkdir -p "$RESULT_ROOT/logs"

nohup bash scripts/run_yelp_all_evals.sh \
  --model-path "$ZERO_SHOT_CKPT" \
  --model-base "$BASE_MODEL" \
  --dataset-root "$GRASPLLM_DATASET_ROOT" \
  --support-root "$FS_ROOT" \
  --output-dir "$RESULT_ROOT" \
  --gpus 0,1 \
  --max-validation-queries 1000 \
  --icl-only \
  > "$RESULT_ROOT/logs/icl_1_5_10.log" 2>&1 &

echo $! | tee "$RESULT_ROOT/logs/icl_1_5_10.pid"
tail -f "$RESULT_ROOT/logs/icl_1_5_10.log"
```

如果只有一张 GPU，把两条命令中的 `--gpus 0,1` 改成 `--gpus 0`，RUR/RBR 会顺序执行。

`--max-validation-queries 1000` 只对 validation 使用 seed 42 做确定性分层抽样，并保持原标签比例；完整 test 不会缩减。省略该参数时仍使用全部 validation。不要用旧参数 `--max-queries` 代替，因为它会同时限制 validation 和 test。

## 8. 结果汇总与检查

统一脚本结束后会自动汇总。也可以单独重新汇总：

```bash
cd "$REPO"

python -m experiments.yelpzip_fewshot.summarize \
  --root "$RESULT_ROOT"
```

查看总表：

```bash
cat "$RESULT_ROOT/all_results_summary.md"
```

主要输出：

```text
$RESULT_ROOT/all_results_summary.md
$RESULT_ROOT/all_results_summary.json
$RESULT_ROOT/zero_shot/yelpzip_rur/probability_metrics.json
$RESULT_ROOT/zero_shot/yelpzip_rbr/probability_metrics.json
$RESULT_ROOT/few_shot/yelpzip_rur/k{1,5,10}_seed{42,43,44,45,46}/probability_metrics.json
$RESULT_ROOT/few_shot/yelpzip_rbr/k{1,5,10}_seed{42,43,44,45,46}/probability_metrics.json
```

`probability_metrics.json` 保存验证集选择的阈值及测试集 PR-AUC、ROC-AUC、Fraud F1、Precision、Recall 和 Balanced Accuracy；`test_predictions.jsonl` 保存逐评论欺诈概率。

## 9. 最短执行顺序

已有原始 YelpZip CSV 和官方源数据集时，完整顺序为：

```text
YelpZip CPU 预处理
→ Stage 0：YelpZip 和 Stage-1 源数据集 embedding
→ Stage 1：单卡训练 Motif-GNN
→ Stage 2：生成 Arxiv、Yelp RUR、Yelp RBR OCS
→ Stage 3：只在 Arxiv 上训练 Projector
→ CPU 生成 1/5/10-shot support
→ GPU 运行 zero-shot 与 ICL
→ 汇总 RUR/RBR 结果
```
