# YelpZip GraspLLM 推理命令

本文只运行推理，不重新执行 Stage 0、Stage 1、Stage 2 或 Stage 3，也不会训练 GNN、Projector 或 LLM。默认使用已经生成的 YelpZip 数据、OCS 上下文、Arxiv 对齐 checkpoint 和 1/5/10-shot support 文件。

## 1. 进入仓库并配置路径

```bash
conda activate graspllm

export REPO=/home/xiewt24/liuyc/GraspLLM-Yelp
export GRASPLLM_DATASET_ROOT=$REPO/dataset
export GRASPLLM_CHECKPOINT_ROOT=$REPO/graspllm_checkpoints
export BASE_MODEL=/data/Qwen3-8B/Qwen3-8B
export ZERO_SHOT_CKPT=$GRASPLLM_CHECKPOINT_ROOT/grasp-qwen3-arxiv-zero-shot
export FS_ROOT=$REPO/artifacts/yelpzip_original_fewshot
export RESULT_ROOT=$REPO/output
export PYTHONPATH=$REPO
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$RESULT_ROOT/logs"
```

如果 Qwen3-8B 不在上述位置，先查找实际模型目录：

```bash
find /data -name config.json -path '*Qwen3-8B*' -print 2>/dev/null
```

然后把 `BASE_MODEL` 修改为包含 `config.json` 的目录。

## 2. 推理前检查

```bash
echo "REPO=$REPO"
echo "DATASET=$GRASPLLM_DATASET_ROOT"
echo "CHECKPOINT=$ZERO_SHOT_CKPT"
echo "BASE_MODEL=$BASE_MODEL"
echo "SUPPORT=$FS_ROOT"

test -f "$BASE_MODEL/config.json" && echo '[OK] Qwen3-8B' || echo '[MISSING] Qwen3-8B'
test -d "$ZERO_SHOT_CKPT" && echo '[OK] Arxiv projector checkpoint' || echo '[MISSING] checkpoint'

for relation in yelpzip_rur yelpzip_rbr; do
  for file in processed_data.pt qwen3_emb_x.pt ocs_val.jsonl ocs_test.jsonl; do
    path="$GRASPLLM_DATASET_ROOT/$relation/$file"
    test -f "$path" && echo "[OK] $path" || echo "[MISSING] $path"
  done
done

for relation in yelpzip_rur yelpzip_rbr; do
  for k in 1 5 10; do
    for seed in 42 43 44 45 46; do
      support_dir="$FS_ROOT/$relation/k${k}_seed${seed}"
      for file in support_train.jsonl validation_holdout.jsonl support_manifest.json; do
        test -f "$support_dir/$file" || echo "[MISSING] $support_dir/$file"
      done
    done
  done
done

nvidia-smi
```

只有在没有出现 `[MISSING]` 时才开始正式推理。

## 3. 推荐命令：只运行 1/5/10-shot ICL

如果 zero-shot 已经完成，并且结果位于：

```text
$RESULT_ROOT/zero_shot/yelpzip_rur/probability_metrics.json
$RESULT_ROOT/zero_shot/yelpzip_rbr/probability_metrics.json
```

则使用下面的命令。GPU 0 运行 RUR，GPU 1 运行 RBR；模型、LLM、Projector 和 GNN 在推理期间全部冻结。

```bash
cd "$REPO"

nohup bash scripts/run_yelp_all_evals.sh \
  --model-path "$ZERO_SHOT_CKPT" \
  --model-base "$BASE_MODEL" \
  --dataset-root "$GRASPLLM_DATASET_ROOT" \
  --support-root "$FS_ROOT" \
  --output-dir "$RESULT_ROOT" \
  --gpus 0,1 \
  --icl-only \
  > "$RESULT_ROOT/logs/icl_1_5_10.log" 2>&1 &

echo $! | tee "$RESULT_ROOT/logs/icl_1_5_10.pid"
```

查看进度：

```bash
tail -f "$RESULT_ROOT/logs/icl_1_5_10.log"
```

退出 `tail -f` 使用 `Ctrl+C`，不会终止后台推理。查看后台进程：

```bash
cat "$RESULT_ROOT/logs/icl_1_5_10.pid"
ps -fp "$(cat "$RESULT_ROOT/logs/icl_1_5_10.pid")"
nvidia-smi
```

## 4. 可选命令：重新运行 zero-shot 和全部 ICL

如果需要从头运行 zero-shot、1-shot、5-shot 和 10-shot，去掉 `--icl-only`：

```bash
cd "$REPO"

nohup bash scripts/run_yelp_all_evals.sh \
  --model-path "$ZERO_SHOT_CKPT" \
  --model-base "$BASE_MODEL" \
  --dataset-root "$GRASPLLM_DATASET_ROOT" \
  --support-root "$FS_ROOT" \
  --output-dir "$RESULT_ROOT" \
  --gpus 0,1 \
  > "$RESULT_ROOT/logs/zero_and_icl_all.log" 2>&1 &

echo $! | tee "$RESULT_ROOT/logs/zero_and_icl_all.pid"
tail -f "$RESULT_ROOT/logs/zero_and_icl_all.log"
```

不要同时运行第 3 节和第 4 节的命令，否则两个任务会写入相同结果目录。

## 5. 只重新汇总已有结果

```bash
cd "$REPO"

python -m experiments.yelpzip_fewshot.summarize \
  --root "$RESULT_ROOT"
```

最终汇总文件为：

```text
$RESULT_ROOT/all_results_summary.md
$RESULT_ROOT/all_results_summary.json
```

各实验的详细结果位于：

```text
$RESULT_ROOT/zero_shot/yelpzip_rur/
$RESULT_ROOT/zero_shot/yelpzip_rbr/
$RESULT_ROOT/few_shot/yelpzip_rur/k{1,5,10}_seed{42,43,44,45,46}/
$RESULT_ROOT/few_shot/yelpzip_rbr/k{1,5,10}_seed{42,43,44,45,46}/
```

每个目录中的 `probability_metrics.json` 保存验证集阈值、测试集 PR-AUC、ROC-AUC、Fraud F1、Precision、Recall 和 Balanced Accuracy；`test_predictions.jsonl` 保存逐评论欺诈概率。
