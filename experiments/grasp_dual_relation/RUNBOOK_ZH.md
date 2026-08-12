# GraspLLM 双关系时间因果实验运行手册

以下命令均从 GraspLLM 仓库根目录运行。Windows 本地负责 Gate 0/1；文本编码、GNN、Projector 和 LLM 评分建议在 Linux CUDA 主机运行。

## 1. Windows：Gate 0/1

```powershell
$Repo = 'C:\Users\31872\Desktop\project\graph+llm\project\test\GraspLLM'
$Raw = 'C:\Users\31872\Desktop\project\graph+llm\source_data\Yelp-Dataset\yelpzip.csv'
$Py = 'E:\anaconda\envs\gnn\python.exe'
$Art = Join-Path $Repo 'artifacts\grasp_dual_relation\yelpzip'
Set-Location $Repo
$env:PYTHONPATH = $Repo

& $Py -m pytest experiments\grasp_dual_relation\tests -q
& $Py -m experiments.grasp_dual_relation.cli --raw-path $Raw --output-dir $Art
```

只有测试全部通过，且 `leakage_audit.json` 中 `passed=true`，才能进入 GPU 阶段。

## 2. Linux GPU：统一变量与环境检查

```bash
export REPO=/abs/path/to/GraspLLM
export RAW=/abs/path/to/yelpzip.csv
export EMBED_MODEL=/data/Qwen3-Embedding-8B
export BASE_MODEL=/data/Qwen/Qwen3-8B
export GRASP_DUAL_EMBED_MODEL="$EMBED_MODEL"
export GRASP_DUAL_BASE_MODEL="$BASE_MODEL"
export ART=$REPO/artifacts/grasp_dual_relation/yelpzip
export GNN=$REPO/checkpoints/grasp_dual_relation/yelpzip
export CTX=$ART/contexts_pilot
export PRED=$ART/predictions_pilot
export PYTHONPATH=$REPO
export CUDA_VISIBLE_DEVICES=0
cd "$REPO"

python -c "import torch, transformers, torch_geometric; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
python -m pytest experiments/grasp_dual_relation/tests -q
```

若尚未安装环境：

```bash
conda create -n graspllm python=3.12 -y
conda activate graspllm
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install torch_geometric==2.6.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

## 3. Gate 0：完整数据准备与文本编码

```bash
python -u -m experiments.grasp_dual_relation.cli \
  --raw-path "$RAW" --output-dir "$ART"

python -u -m experiments.grasp_dual_relation.encode_text \
  --raw-path "$RAW" \
  --model-path "$EMBED_MODEL" \
  --output "$ART/qwen3_emb_x.pt" \
  --max-length 512 --batch-size 4 --device cuda
```

如果输出的 `truncation_sample_rate > 0.05`，删除该 embedding 后把 `--max-length` 改为 `1024` 重跑；若明确覆盖旧文件，可加 `--overwrite`。

## 4. Stage 1：双 GNN 自监督预训练与冻结编码

```bash
python -u -m experiments.grasp_dual_relation.pretrain_gnn \
  --graph-bundle "$ART/temporal_graphs.npz" \
  --embedding "$ART/qwen3_emb_x.pt" \
  --output-dir "$GNN" \
  --relations user business \
  --epochs 300 --max-nodes 4000 --lr 1e-4 --seed 42 --device cuda

python -u -m experiments.grasp_dual_relation.infer_gnn \
  --graph-bundle "$ART/temporal_graphs.npz" \
  --embedding "$ART/qwen3_emb_x.pt" \
  --checkpoint-dir "$GNN" --output-dir "$GNN" \
  --relations user business --batch-size 256 --device cuda
```

应得到 `user_structure_learner.pth`、`business_structure_learner.pth`、`user_structure_emb.pt`、`business_structure_emb.pt`。

## 5. Gate 2：2,000 条验证查询 smoke

```bash
python -u -m experiments.grasp_dual_relation.generate_contexts \
  --raw-path "$RAW" --graph-bundle "$ART/temporal_graphs.npz" \
  --text-embedding "$ART/qwen3_emb_x.pt" \
  --user-structure-embedding "$GNN/user_structure_emb.pt" \
  --business-structure-embedding "$GNN/business_structure_emb.pt" \
  --output-dir "$ART/contexts_smoke" \
  --variant dual --method ocs --splits validation \
  --query-sample-size 2000 --max-depth 1 \
  --user-k 8 --business-k 8 --beta-user 0.55 --beta-business 0.55 --seed 42
```

用下节的训练与评价命令时，将数据路径换成 smoke 文件，并添加 `--max-queries 20`，先确认无 OOM、NaN 和接口错误。

## 6. Gate 3：生成全量对齐/验证和固定 20,000 测试矩阵

```bash
mkdir -p "$CTX"
for spec in \
  "dual ocs" \
  "user_only ocs" \
  "business_only ocs" \
  "merged ocs" \
  "dual random" \
  "dual text_topk" \
  "text_only ocs"
do
  read -r variant method <<< "$spec"
  python -u -m experiments.grasp_dual_relation.generate_contexts \
    --raw-path "$RAW" --graph-bundle "$ART/temporal_graphs.npz" \
    --text-embedding "$ART/qwen3_emb_x.pt" \
    --user-structure-embedding "$GNN/user_structure_emb.pt" \
    --business-structure-embedding "$GNN/business_structure_emb.pt" \
    --output-dir "$CTX" --variant "$variant" --method "$method" \
    --splits alignment validation --max-depth 1 \
    --user-k 8 --business-k 8 --merged-k 16 \
    --beta-user 0.55 --beta-business 0.55 --seed 42

  python -u -m experiments.grasp_dual_relation.generate_contexts \
    --raw-path "$RAW" --graph-bundle "$ART/temporal_graphs.npz" \
    --text-embedding "$ART/qwen3_emb_x.pt" \
    --user-structure-embedding "$GNN/user_structure_emb.pt" \
    --business-structure-embedding "$GNN/business_structure_emb.pt" \
    --output-dir "$CTX" --variant "$variant" --method "$method" \
    --splits test --query-sample-size 20000 --max-depth 1 \
    --user-k 8 --business-k 8 --merged-k 16 \
    --beta-user 0.55 --beta-business 0.55 --seed 42
done
```

文件前缀映射：完整双图 OCS=`ocs`，用户=`user_only_ocs`，商家=`business_only_ocs`，合并图=`merged_ocs`，随机=`random`，文本 Top-K=`text_topk`，纯文本=`text_only_ocs`。

## 7. 全量 60%–70% 对齐上界

训练六个图模型；基础 LLM 与 embedding 均冻结：

```bash
train_graph_variant () {
  name=$1
  prefix=$2
  dual=$3
  DATA_PATH="$CTX/${prefix}_alignment.jsonl" \
  GRAPH_EMBEDDING="$ART/qwen3_emb_x.pt" \
  BASE_MODEL="$BASE_MODEL" \
  OUTPUT_DIR="$GNN/projectors/grasp_${name}" \
  DUAL_GRAPH_PROJECTOR="$dual" \
  EPOCHS=1 MAX_STEPS=-1 BATCH_SIZE=1 GRAD_ACCUM=16 LR=5e-4 GPU=0 \
  bash scripts/stage3_dual_projector_train.sh
}

train_graph_variant dual_ocs ocs True
train_graph_variant user_ocs user_only_ocs True
train_graph_variant business_ocs business_only_ocs True
train_graph_variant merged_ocs merged_ocs False
train_graph_variant dual_random random True
train_graph_variant dual_text_topk text_topk True
```

评价并在验证集选择阈值：

```bash
mkdir -p "$PRED"
eval_graph_variant () {
  name=$1
  prefix=$2
  for split in validation test
  do
    python -u -m experiments.grasp_dual_relation.evaluate_llm \
      --model-path "$GNN/projectors/grasp_${name}" \
      --model-base "$BASE_MODEL" \
      --data-path "$CTX/${prefix}_${split}.jsonl" \
      --graph-embedding "$ART/qwen3_emb_x.pt" \
      --answers-file "$PRED/${name}_${split}.jsonl" --device cuda
  done
  python -m experiments.grasp_dual_relation.score_results \
    --validation "$PRED/${name}_validation.jsonl" \
    --test "$PRED/${name}_test.jsonl" \
    --output "$PRED/${name}_metrics.json"
}

eval_graph_variant dual_ocs ocs
eval_graph_variant user_ocs user_only_ocs
eval_graph_variant business_ocs business_only_ocs
eval_graph_variant merged_ocs merged_ocs
eval_graph_variant dual_random random
eval_graph_variant dual_text_topk text_topk

for split in validation test
do
  python -u -m experiments.grasp_dual_relation.evaluate_llm \
    --text-only --model-base "$BASE_MODEL" \
    --data-path "$CTX/text_only_ocs_${split}.jsonl" \
    --answers-file "$PRED/text_only_${split}.jsonl" --device cuda
done
python -m experiments.grasp_dual_relation.score_results \
  --validation "$PRED/text_only_validation.jsonl" \
  --test "$PRED/text_only_test.jsonl" \
  --output "$PRED/text_only_metrics.json"
```

## 8. 严格 few-shot：K=1/5/10，seed=42–46

下面对核心三方比较（完整双图、合并图、纯文本）运行全部 15 组。其他结构消融可用相同方式替换前缀。

```bash
export FS=$ART/fewshot_pilot
mkdir -p "$FS" "$PRED/fewshot"

for k in 1 5 10
do
  for seed in 42 43 44 45 46
  do
    for prefix in ocs merged_ocs text_only_ocs
    do
      for split in validation test
      do
        out="$FS/${prefix}/k${k}_seed${seed}/${split}"
        python -m experiments.grasp_dual_relation.make_fewshot_data \
          --alignment-contexts "$CTX/${prefix}_alignment.jsonl" \
          --query-contexts "$CTX/${prefix}_${split}.jsonl" \
          --support-ids "$ART/support_ids.json" \
          --output-dir "$out" --shots "$k" --seed "$seed"
      done
    done

    DATA_PATH="$FS/ocs/k${k}_seed${seed}/validation/k${k}_seed${seed}_train.jsonl" \
    GRAPH_EMBEDDING="$ART/qwen3_emb_x.pt" BASE_MODEL="$BASE_MODEL" \
    OUTPUT_DIR="$GNN/projectors/grasp_dual_ocs_k${k}_seed${seed}" \
    DUAL_GRAPH_PROJECTOR=True MAX_STEPS=200 EPOCHS=1 GPU=0 \
    bash scripts/stage3_dual_projector_train.sh

    DATA_PATH="$FS/merged_ocs/k${k}_seed${seed}/validation/k${k}_seed${seed}_train.jsonl" \
    GRAPH_EMBEDDING="$ART/qwen3_emb_x.pt" BASE_MODEL="$BASE_MODEL" \
    OUTPUT_DIR="$GNN/projectors/grasp_merged_ocs_k${k}_seed${seed}" \
    DUAL_GRAPH_PROJECTOR=False MAX_STEPS=200 EPOCHS=1 GPU=0 \
    bash scripts/stage3_dual_projector_train.sh

    for split in validation test
    do
      python -u -m experiments.grasp_dual_relation.evaluate_llm \
        --model-path "$GNN/projectors/grasp_dual_ocs_k${k}_seed${seed}" \
        --model-base "$BASE_MODEL" \
        --data-path "$FS/ocs/k${k}_seed${seed}/${split}/k${k}_seed${seed}_eval.jsonl" \
        --graph-embedding "$ART/qwen3_emb_x.pt" \
        --answers-file "$PRED/fewshot/dual_k${k}_seed${seed}_${split}.jsonl" --device cuda

      python -u -m experiments.grasp_dual_relation.evaluate_llm \
        --model-path "$GNN/projectors/grasp_merged_ocs_k${k}_seed${seed}" \
        --model-base "$BASE_MODEL" \
        --data-path "$FS/merged_ocs/k${k}_seed${seed}/${split}/k${k}_seed${seed}_eval.jsonl" \
        --graph-embedding "$ART/qwen3_emb_x.pt" \
        --answers-file "$PRED/fewshot/merged_k${k}_seed${seed}_${split}.jsonl" --device cuda

      python -u -m experiments.grasp_dual_relation.evaluate_llm \
        --text-only --model-base "$BASE_MODEL" \
        --data-path "$FS/text_only_ocs/k${k}_seed${seed}/${split}/k${k}_seed${seed}_eval.jsonl" \
        --answers-file "$PRED/fewshot/text_k${k}_seed${seed}_${split}.jsonl" --device cuda
    done

    for name in dual merged text
    do
      python -m experiments.grasp_dual_relation.score_results \
        --validation "$PRED/fewshot/${name}_k${k}_seed${seed}_validation.jsonl" \
        --test "$PRED/fewshot/${name}_k${k}_seed${seed}_test.jsonl" \
        --output "$PRED/fewshot/${name}_k${k}_seed${seed}_metrics.json"
    done

    python -m experiments.grasp_dual_relation.compare_results \
      --candidate "$PRED/fewshot/dual_k${k}_seed${seed}_test.jsonl" \
      --baseline "text=$PRED/fewshot/text_k${k}_seed${seed}_test.jsonl" \
      --baseline "merged=$PRED/fewshot/merged_k${k}_seed${seed}_test.jsonl" \
      --bootstrap-samples 2000 --seed "$seed" --minimum-delta 0.01 \
      --output "$PRED/fewshot/dual_k${k}_seed${seed}_bootstrap.json"
  done

  python -m experiments.grasp_dual_relation.aggregate_results \
    --inputs \
      "$PRED/fewshot/dual_k${k}_seed42_metrics.json" \
      "$PRED/fewshot/dual_k${k}_seed43_metrics.json" \
      "$PRED/fewshot/dual_k${k}_seed44_metrics.json" \
      "$PRED/fewshot/dual_k${k}_seed45_metrics.json" \
      "$PRED/fewshot/dual_k${k}_seed46_metrics.json" \
    --output "$PRED/fewshot/dual_k${k}_summary.json"
done
```

## 9. 扩展到完整测试集

只有固定 20,000 测试集上的完整双图相对纯文本和合并图均满足：`delta_pr_auc >= 0.01`、同时置信区间下界大于 0、Holm 校正后 `p < 0.05`，才继续。

```bash
export CTX_FULL=$ART/contexts_full
mkdir -p "$CTX_FULL"
for spec in "dual ocs" "merged ocs" "text_only ocs"
do
  read -r variant method <<< "$spec"
  python -u -m experiments.grasp_dual_relation.generate_contexts \
    --raw-path "$RAW" --graph-bundle "$ART/temporal_graphs.npz" \
    --text-embedding "$ART/qwen3_emb_x.pt" \
    --user-structure-embedding "$GNN/user_structure_emb.pt" \
    --business-structure-embedding "$GNN/business_structure_emb.pt" \
    --output-dir "$CTX_FULL" --variant "$variant" --method "$method" \
    --splits test --max-depth 1 --user-k 8 --business-k 8 --merged-k 16 \
    --beta-user 0.55 --beta-business 0.55 --seed 42
done
```

随后将第 8 节中的测试查询路径从 `$CTX` 换为 `$CTX_FULL`，重新构造 few-shot test JSONL、评价和统计；不要重新选择阈值，必须复用对应 seed 验证集产生的阈值。

如果 Gate 失败，停止增加 DFS 深度、shot、motif 或 LLM 微调，并报告核心假设暂未得到支持。
