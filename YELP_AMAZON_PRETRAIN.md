# Yelp + Amazon 预训练与云端运行手册

该流程使用 YelpZip 和六个 Amazon 评论类目替换原来的 Stage 1 源数据，并在
Yelp/Amazon 监督 OCS 上训练 projector。由于 projector 会看到 Yelp 训练分割标签，
最终 Yelp 推理称为 `no_icl`，不是 zero-shot。

## 1. 本地生成上传数据包

原始数据根目录必须包含：

```text
source_data/
├── Yelp-Dataset/yelpzip.csv
└── amazon/
    ├── Cell_Phones_and_Accessories.json
    ├── Clothing_Shoes_and_Jewelry.json
    ├── Electronics.json
    ├── Home_and_Kitchen.json
    ├── Sports_and_Outdoors.json
    └── Toys_and_Games.json
```

`part.json` 和 `separate.json` 是派生文件，不参与抽样。Windows 本地执行：

```powershell
cd C:\Users\31872\Desktop\project\graph+llm\project\test\GraspLLM-Yelp-publish

E:\anaconda\envs\gnn\python.exe scripts\build_yelp_amazon_upload_bundle.py `
  --source-root C:\Users\31872\Desktop\project\graph+llm\source_data `
  --output-root C:\Users\31872\Desktop\project\graph+llm\project\test\GraspLLM-Yelp-publish\artifacts\training_data\yelp_amazon_pretrain_raw_v1 `
  --amazon-per-category 100000 `
  --seed 42 `
  --part-size-gib 2 `
  --zstd-level 10
```

输出包括六个新 Amazon JSONL、完整 YelpZip CSV、`MANIFEST.json`、
`SHA256SUMS`、云端说明和最多 2 GiB 的 `.tar.zst.part-*` 分卷。Amazon 每个
JSONL 恰好 100,000 条，`label=0/1` 各 50,000 条。

## 2. 云端合并与解压

```bash
cd /path/to/uploaded/yelp_amazon_pretrain_raw_v1

cat parts/yelp_amazon_pretrain_raw_v1.tar.zst.part-* \
  > yelp_amazon_pretrain_raw_v1.tar.zst

sha256sum -c SHA256SUMS
zstd -dc yelp_amazon_pretrain_raw_v1.tar.zst | tar -xf -
```

解压后训练入口是：

```text
/path/to/uploaded/yelp_amazon_pretrain_raw_v1/yelp_amazon_pretrain_raw_v1/raw_data
```

## 3. 环境变量

```bash
export REPO=/home/xiewt24/liuyc/GraspLLM-Yelp
export RAW_REVIEW_ROOT=/path/to/uploaded/yelp_amazon_pretrain_raw_v1/yelp_amazon_pretrain_raw_v1/raw_data
export GRASPLLM_DATASET_ROOT=$REPO/dataset
export GRASPLLM_MODELS_ROOT=$REPO/models
export GRASPLLM_CHECKPOINT_ROOT=$REPO/graspllm_checkpoints
export GRASPLLM_QWEN3_EMB_MODEL=$GRASPLLM_MODELS_ROOT/Qwen3-Embedding-8B
export BASE_MODEL=$GRASPLLM_MODELS_ROOT/Qwen3-8B

cd "$REPO"
```

默认 GPU 分配：Stage 0 使用 `0,1,2,3` 分片；Stage 1 使用 GPU 0；Stage 2
依次让每个关系图使用四卡 motif-parallel；Stage 3 使用四卡 DDP；最终 RUR/RBR
评估使用 GPU 0/1。可通过 `EMBED_GPUS`、`STAGE1_GPU`、`STAGE2_GPUS`、
`STAGE2_LOGICAL_GPUS`、`STAGE3_GPUS` 和 `EVAL_GPUS` 覆盖。

## 4. 分阶段正式命令

建议逐阶段前台执行，成功后再进入下一阶段：

```bash
bash scripts/run_yelp_amazon_pretrain.sh prepare
bash scripts/run_yelp_amazon_pretrain.sh embed
bash scripts/run_yelp_amazon_pretrain.sh stage1
bash scripts/run_yelp_amazon_pretrain.sh stage2
bash scripts/run_yelp_amazon_pretrain.sh stage3
bash scripts/run_yelp_amazon_pretrain.sh support
bash scripts/run_yelp_amazon_pretrain.sh eval
```

`prepare` 生成以下 14 个图：

```text
yelpzip_rur yelpzip_rbr
amazon_cellphones_rur amazon_cellphones_rpr
amazon_clothing_rur amazon_clothing_rpr
amazon_electronics_rur amazon_electronics_rpr
amazon_home_rur amazon_home_rpr
amazon_sports_rur amazon_sports_rpr
amazon_toys_rur amazon_toys_rpr
```

其中 `rur` 表示 review-user-review，Amazon `rpr` 表示
review-product-review。图构造不读取标签，最大度为 32。

`embed` 只编码 Yelp 和六个 Amazon 类目的七份唯一文本集合，随后经
`review_id_hash`、`text_hash`、`mask_hash` 验证，把 embedding 复用于第二关系图。

`stage1` 默认运行 300 epochs，每个数据集每轮一个 2,000 节点子图。完整
embedding 以 FP16 留在 CPU，共享文本集合只缓存一次；GPU 只接收当前子图。

`stage2` 对 Yelp 两个关系各生成 60,000 条训练 OCS，对十二个 Amazon 关系图
各生成 10,000 条；每份均为 `Legitimate/Fraudulent=1:1`。Yelp 训练 OCS 的
中心和所有上下文节点必须属于 train mask，否则立即失败。

`stage3` 使用 Qwen3-8B、四张 V100、FP16、每卡 batch 1、gradient
accumulation 2、1 epoch 和 2,048 token 训练上限，仅训练 projector。输出：

```text
$GRASPLLM_CHECKPOINT_ROOT/grasp-qwen3-yelp-amazon-projector/mm_projector.bin
```

`eval` 保持完整 Yelp 查询/support 文本和原生 32K 推理上下文，输出根目录：

```text
$REPO/artifacts/yelpzip_yelp_amazon_results/
├── no_icl/
└── few_shot/
```

## 5. 隔离和审计产物

- Yelp 固定 10,000 validation、20,000 test，其余节点为 train。
- Stage 1 只使用 Yelp train-induced edges。
- Stage 3 的 Yelp OCS 中，中心节点和图上下文均不得进入 validation/test。
- validation 仅用于阈值选择；test 仅用于最终 PR-AUC、ROC-AUC、Fraud F1、
  Precision、Recall 和 Balanced Accuracy。
- 每一阶段会写出：

```text
$GRASPLLM_DATASET_ROOT/yelp_amazon_prepared_audit.json
$GRASPLLM_DATASET_ROOT/yelp_amazon_embedding_audit.json
$GRASPLLM_DATASET_ROOT/yelp_amazon_ocs_leakage_audit.json
```

若需要完全重建已有 `processed_data.pt`，显式设置：

```bash
FORCE=1 bash scripts/run_yelp_amazon_pretrain.sh prepare
```

该选项只清理这 14 个数据集目录中的派生产物，不触碰上传的原始数据包。
