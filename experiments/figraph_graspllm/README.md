# FiGraph × GraspLLM strict transfer

This package implements an isolated, auditable FiGraph experiment family. It does not alter the YelpZip protocol.

## Dataset provisioning

FiGraph payload files are intentionally excluded from Git and Git LFS. Obtain them separately (for example, by uploading them to the target Linux server) and place them under `dataset/FiGraph/data/` before running an experiment. The expected annual-file layout is described by the upstream [FiGraph repository](https://github.com/XiaoguangWang23/FiGraph).

## Fixed contracts

- Company-year node key, annual direct-company graphs, and a 2014--2022 snapshot disjoint union by default.
- 2019 support only; 2020 threshold/validation; 2021 and 2022 future test.
- Missing MDA remains a zero-text graph node but is excluded from the main cohort.
- Frozen non-FiGraph MotifGNN and frozen non-FiGraph projector. Projector loading requires a hash-bound provenance declaration.
- Native 32,768-token Qwen3 context and `enable_thinking=False`; OOM fallback is 24,576 then 16,384 and is recorded.
- Canonical `Fraud`/`Normal` answers scored by length-normalized likelihood.

## Entry points

- `prepare`: graph/text artifacts and data audits.
- `audit_tokens`: yearly tokenizer distribution before model execution.
- `encode_text`: Qwen3-Embedding-8B, batch size 1, token-level head/middle/tail extraction.
- `support`: deterministic balanced K-per-class manifests.
- `contexts`: frozen MotifGNN inference and 32-node OCS/random contexts.
- `evaluate_llm`: Text-only Matched, Text-only MaxContext, Random Graph + LLM, and Full GraspLLM.
- `frozen_lr`: Frozen MotifGNN + LR.
- `score_results`, `aggregate`, `summarize`, and `gate`: validation-only thresholding, metrics, seed aggregation, and the pre-registered Gate.

Create the projector declaration only after checking its training source:

```bash
python -m experiments.figraph_graspllm.make_projector_provenance \
  --model-path /path/to/projector \
  --source-datasets arxiv pubmed computer history reddit \
  --confirm-no-figraph-training \
  --output artifacts/figraph/projector_provenance.json
```

Then export the paths required at the top of `scripts/run_figraph_mvp.sh` and select `MODE=smoke`, `MODE=pilot`, or `MODE=full`. Full mode refuses to start unless the pilot Gate JSON says `PASS`.

If a raw annual payload is unavailable, `FIGRAPH_YEARS` can explicitly select the snapshots to process. It must retain 2019--2022, and the output manifest records the omitted years. For example, an unavailable 2014 workbook can be excluded with `FIGRAPH_YEARS="2015 2016 2017 2018 2019 2020 2021 2022"`; this is an eight-snapshot protocol variant, not the default nine-snapshot experiment.

## V100 profiles

`scripts/run_figraph_v100x2.sh` and `scripts/run_figraph_v100x4.sh` support two or four 32GB V100 cards through process-level data parallelism. They deliberately use FP16 + SDPA and fix the context at 16,384 tokens because V100 does not support the Ampere BF16/FlashAttention-2 path. Both profiles:

- encode one deterministic node shard per GPU and refuse to merge shards with different context lengths or provenance;
- runs at most one Qwen3 process per GPU;
- schedule contexts and independent method/seed/split evaluations in two- or four-process batches;
- write results below separate `v100x2_fp16_16384` or `v100x4_fp16_16384` artifact namespaces, so neither can be mistaken for the native-32K primary experiment.

Two-card example:

```bash
V100_GPU_IDS=0,1 MAX_LENGTH=16384 MODE=smoke \
  bash scripts/run_figraph_v100x2.sh
```

Four-card example:

```bash
V100_GPU_IDS=0,1,2,3 MAX_LENGTH=16384 MODE=smoke \
  bash scripts/run_figraph_v100x4.sh
```

Both profiles require 32GB V100 cards. A 16GB V100 cannot hold an independent FP16 Qwen3-8B process and is intentionally rejected. The two-card profile preserves the same data, split, support, scoring, and Gate contracts as the four-card profile; it only reduces process-level concurrency and uses two embedding shards.
