# YelpZip random difficulty baseline

该实验用于给双关系 GraspLLM 提供严格可比的随机难度下界。它复用
`grasp_dual_relation` 生成的时间 split，不重新划分数据。

- `uniform`：与标签无关的连续均匀随机分数，使用 seed 42–46。
- `alignment_prior`：仅使用历史 alignment 集欺诈率的常数分数。
- 阈值仅由 validation 集选择，test 标签只用于最终指标。
- 主参照为随机排序的期望 `ROC-AUC=0.5`、`PR-AUC=测试集欺诈率`。

完整测试：

```bash
python -m experiments.random.run \
  --raw-path /path/to/yelpzip.csv \
  --graph-bundle artifacts/grasp_dual_relation/yelpzip/temporal_graphs.npz \
  --output-dir artifacts/random/yelpzip \
  --methods uniform alignment_prior \
  --seeds 42 43 44 45 46
```

固定 20,000 条、按标签和时间分层的测试集：

```bash
python -m experiments.random.run \
  --raw-path /path/to/yelpzip.csv \
  --graph-bundle artifacts/grasp_dual_relation/yelpzip/temporal_graphs.npz \
  --output-dir artifacts/random/yelpzip_pilot20k \
  --test-sample-size 20000 --sample-seed 42
```

输出包括逐查询概率、各 seed 指标以及 `run_manifest.json`。随机基线与模型比较时，
必须确认 `node_ids_hash` 或预测文件中的 `node_id` 完全一致。

与相同测试节点上的模型预测做配对 bootstrap：

```bash
python -m experiments.grasp_dual_relation.compare_results \
  --candidate artifacts/grasp_dual_relation/yelpzip/predictions_pilot/dual_ocs_test.jsonl \
  --baseline random=artifacts/random/yelpzip_pilot20k/uniform_seed42_test.jsonl \
  --output artifacts/random/yelpzip_pilot20k/model_vs_random.json
```

如果模型的 PR-AUC 只接近 `test_fraud_prevalence`、ROC-AUC 只接近 0.5，说明模型对
未来欺诈评论的排序能力与随机排序接近。F1 较高不一定代表可区分性强，因为验证集选出的
低阈值可能使随机方法把大部分样本都预测为欺诈；因此难度结论优先看 PR-AUC 和 ROC-AUC。

原始 GraspLLM 静态 YelpZip base 必须直接复用 `processed_data.pt` 中的固定 mask：

```bash
python -m experiments.random.run_static_yelp \
  --processed-data dataset/yelpzip_rur/processed_data.pt \
  --output-dir artifacts/random/yelpzip_static
```
