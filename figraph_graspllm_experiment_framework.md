# FiGraph × GraspLLM 金融欺诈检测实验框架

## 1. 实验目标

目标是测试 **GraspLLM 在金融领域 Text-Attributed Graph（TAG）欺诈检测任务中的适用性**。

FiGraph 中，每个目标节点表示某一年度的上市公司，公司节点具有：

- 原始 MDA（Management's Discussion and Analysis）金融文本；
- 财务结构化特征；
- 与其他公司或金融实体的关系；
- Fraud / Normal 标签。

本实验第一阶段尽量保持 GraspLLM 原始框架不变，只修改数据预处理与图构建方式。

核心任务定义为：

\[
\boxed{
\text{Company-Year Node}
+
\text{MDA Text}
+
\text{Financial Graph}
\rightarrow
\text{Fraud / Normal}
}
\]

---

## 2. 总体实验流程

```text
FiGraph 原始动态图
        │
        ├── MDA 文本
        ├── 公司标签
        ├── 公司财务属性
        └── 异构金融关系
        │
        ▼
构建 Company-only Homogeneous TAG
        │
        ├── Node = Listed Company-Year
        ├── Text = MDA
        ├── Edge = Direct Company-Company Relation
        └── Label = Fraud / Normal
        │
        ▼
Frozen Text Embedding Model
        │
        ▼
GraspLLM Structure Learner / MotifGNN
        │
        ▼
OCS Subgraph Selection
        │
        ▼
Projector
        │
        ▼
Frozen LLM
        │
        ▼
Fraud / Normal Prediction
```

---

# 3. 数据构建

## 3.1 节点定义

将每一个 **上市公司-年份实例** 定义为一个节点：

\[
v_{i,t} = \text{Company}_i \text{ at Year } t
\]

例如：

```text
Company_A_2018
Company_A_2019
Company_A_2020
```

必须视为不同节点。

不要把同一家公司的多个年份合并，因为同一公司不同年份可能具有不同欺诈状态。

---

## 3.2 节点文本

第一阶段只使用 FiGraph 提供的原始 **MDA 文本**：

\[
x^{text}_{i,t}=MDA_{i,t}
\]

MDA 是公司年报中的管理层讨论与分析，包含：

- 经营情况；
- 收入和利润变化；
- 现金流；
- 资本结构；
- 市场风险；
- 管理层对经营情况的解释；
- 未来经营风险与趋势。

第一阶段不要将结构化财务指标人工转换成文本，以保证：

\[
\boxed{\text{使用真实原始金融文本，而不是人工生成文本}}
\]

---

## 3.3 节点标签

任务为二分类：

\[
y_{i,t}=
\begin{cases}
1,& Fraud\\
0,& Normal
\end{cases}
\]

只对 FiGraph 中有正式 fraud label 的上市公司节点进行监督学习和评价。

---

# 4. Homogeneous TAG 构建

FiGraph 原始数据是 heterogeneous graph。

为了尽量兼容原始 GraspLLM，第一阶段转换为：

\[
G_t=(V_t^C,E_t^C)
\]

其中：

- \(V_t^C\)：年份 \(t\) 的上市公司节点；
- \(E_t^C\)：原始 FiGraph 中两个上市公司之间的直接关系。

---

## 4.1 推荐方案：Direct Company Graph

只保留：

```text
Listed Company ───── Listed Company
```

的原始直接边。

忽略中间 background entities。

例如：

```text
Company A --investment--> Company B
```

转换成：

```text
Company A -------- Company B
```

---

## 4.2 Edge Type 处理

FiGraph 中可能存在：

- Investment
- Related-party Transaction
- Supply Chain
- Audit
- 其他金融关系

第一阶段统一处理为：

\[
A_{ij}=1
\]

即：

> 只要两个公司之间存在至少一种直接关系，就连接一条无向边。

如果同一公司对存在多种关系：

```text
A --Investment--> B
A --SupplyChain--> B
```

仍然只保留：

```text
A -------- B
```

避免 parallel edges 对原始 MotifGNN 造成影响。

---

## 4.3 不推荐：全部异构节点直接 Flatten

不建议第一阶段使用：

```text
Company
Person
Organization
Auditor
Other Entity
```

全部作为 homogeneous node。

原因：

1. background node 通常没有 MDA 文本；
2. 需要人为添加 NULL token；
3. OCS 的 semantic similarity 会受到大量同质 NULL embedding 干扰；
4. 与 GraspLLM 原始 TAG 假设不完全一致。

因此该方案只作为后续消融。

---

## 4.4 不推荐：Naive 2-hop Projection

例如：

```text
Company A
    \
     Auditor X
    /
Company B
```

不要简单转换为：

```text
Company A ----- Company B
```

更不能对连接同一 background node 的所有公司建立 clique。

因为：

\[
d\text{ 个公司共享一个实体}
\]

会生成：

\[
\binom{d}{2}
\]

条人工边。

这会人为制造大量：

- triangles；
- 4-cycles；
- 4-cliques。

而 GraspLLM 恰好依赖 motif，因此可能导致严重的结构偏差。

---

# 5. 时间处理

FiGraph 包含多个年度 snapshot：

```text
2014
2015
...
2022
```

建议把每一年视为独立图：

\[
G_{2014},G_{2015},...,G_{2022}
\]

第一阶段：

- 不建立跨年份 temporal edges；
- 不把同一公司跨年份实例合并；
- 不让未来年份信息参与历史年份预测。

---

# 6. 时间切分方案

推荐采用严格未来预测协议。

## 方案 A：主实验

```text
2014–2018  → Stage-1 Structure Pretraining
2019       → Projector / Support Pool
2020       → Validation
2021–2022  → Test
```

优势：

- 严格时间因果；
- 测试集完全位于未来；
- 与真实金融风险预测场景一致。

---

## 方案 B：滚动验证

为了减少单次时间切分带来的偶然性，可以增加：

```text
Train ≤2018 → Validate 2019 → Test 2020
Train ≤2019 → Validate 2020 → Test 2021
Train ≤2020 → Validate 2021 → Test 2022
```

最后报告三个 future test 的均值和标准差。

该方案适合最终论文实验。

---

# 7. Stage-1：图结构学习

目标：

> 学习金融 company graph 中通用结构模式，不使用测试标签。

输入：

\[
X_v = Encoder(MDA_v)
\]

其中文本编码器冻结。

然后使用 GraspLLM 原始的 structure learner / MotifGNN。

尽量沿用原仓库中的自监督任务和训练策略。

---

## 7.1 Motif

第一阶段保留原始 GraspLLM motif：

- Edge
- Triangle
- 4-cycle
- 4-clique

金融解释：

### Edge

两个公司存在直接金融或商业关系。

### Triangle

三家公司形成闭合关系群：

```text
A ----- B
 \     /
   \ /
    C
```

### 4-cycle

四家公司形成关系闭环：

```text
A --- B
|     |
D --- C
```

### 4-clique

四家公司高度互联。

这些结构可能反映：

- 控股关系；
- 关联交易；
- 供应链团体；
- 高度相关的企业关系网络。

---

# 8. Stage-2：GraphToken 与 LLM 对齐

保持：

\[
\text{Graph Encoder}
\rightarrow
\text{Projector}
\rightarrow
\text{Frozen LLM}
\]

第一阶段实验建议：

- Graph Encoder：冻结；
- LLM：冻结；
- 只训练 Projector。

Projector 可以首先使用两层 MLP：

\[
h_g
\rightarrow
Linear
\rightarrow
GELU
\rightarrow
Linear
\rightarrow
h_{LLM}
\]

目标是把 GraphToken 映射到 LLM embedding space。

---

# 9. OCS 子图选择

Query 为目标公司：

\[
q = Company_{i,t}
\]

候选集合只来自当前年份 snapshot：

\[
\mathcal{N}(q)\subseteq G_t
\]

第一阶段建议尽量保持 GraspLLM 原始 OCS：

\[
Score(v|q)
=
\beta S_{semantic}
+
(1-\beta)S_{structure}
\]

其中：

- Semantic：公司 MDA embedding 与 query MDA embedding 的相关性；
- Structure：MotifGNN representation 的局部结构一致性。

不要第一阶段引入额外金融规则。

---

# 10. Few-shot 设置

每个测试 query 的 demonstrations 从历史 support pool 中选择。

建议测试：

\[
K\in\{1,5,10,16,32\}
\]

其中 K-shot 表示：

```text
K Fraud
+
K Normal
```

如果正样本过少，可优先采用：

```text
1-shot
5-shot
10-shot
16-shot
```

---

## 10.1 Support Pool

例如主时间切分下：

```text
2019 → support pool
```

测试：

```text
2021–2022
```

因此满足：

\[
t_{support}<t_{query}
\]

避免 future leakage。

---

# 11. LLM 输入模板

推荐保持简洁。

例如：

```text
You are given examples of companies and their graph representations.

Example 1:
Company information:
[MDA / optional short text]

Graph representation:
<GraphToken>

Label:
Fraud

Example 2:
...

Target company:
Company information:
[MDA]

Graph representation:
<GraphToken>

Predict whether the target company is:
Fraud or Normal.
```

核心是让：

\[
\text{MDA semantic information}
+
\text{GraphToken structural information}
\]

共同参与 ICL。

---

# 12. Baselines

至少包含以下对照。

## 12.1 Text-only

```text
MDA
↓
LLM
↓
Fraud / Normal
```

用于判断：

> 仅靠财务文本能做到什么程度？

---

## 12.2 Graph-only GNN

```text
MDA embedding
+
Company Graph
↓
GNN
↓
Classifier
```

推荐：

- GCN
- GraphSAGE
- GAT

至少选择一个简单 GNN。

---

## 12.3 GraspLLM without OCS

使用随机或普通邻域：

```text
Random / BFS Subgraph
+
GraphToken
+
LLM
```

用于判断 OCS 是否有效。

---

## 12.4 Random Subgraph + LLM

随机选择相同数量节点作为 context。

用于判断：

\[
\boxed{\text{金融数据本身难度}}
\]

以及 OCS 是否真正选择到了有效 context。

---

## 12.5 Full GraspLLM

```text
MDA
+
MotifGNN
+
OCS
+
GraphToken
+
Frozen LLM
```

作为正式方法。

---

# 13. 关键消融实验

## Ablation A：Graph 信息

```text
Text Only
vs.
Text + Graph
```

回答：

> 公司关系图是否提供额外欺诈信息？

---

## Ablation B：Motif

```text
Edge only
vs.
Edge + Triangle
vs.
Edge + Triangle + 4-cycle
vs.
Full Motif
```

回答：

> 金融欺诈是否真的与高阶结构相关？

---

## Ablation C：OCS

```text
Random
vs.
BFS / Neighbor
vs.
OCS
```

回答：

> GraspLLM 的子图选择模块是否有效？

---

## Ablation D：Graph Construction

比较：

### A. Direct Company Graph

推荐主方案。

### B. All-node Flatten Graph

保留所有 background entities，缺失文本使用 NULL feature。

### C. Degree-controlled 2-hop Projection

只对低度 background nodes 做二跳 company projection：

\[
deg(b)\le d_{max}
\]

例如：

```text
d_max = 5 / 10 / 20
```

用于验证：

> 异构 background relation 是否有帮助？

---

## Ablation E：文本信息

```text
MDA only
vs.
Graph only
vs.
MDA + Graph
```

这是整个实验最重要的消融之一。

---

## Ablation F：结构化财务指标

第一阶段：

```text
MDA only
```

后续增加：

```text
MDA + Financial Features
```

用于判断财务数值属性是否进一步提升性能。

---

# 14. 评价指标

FiGraph 欺诈节点高度不平衡，因此主指标建议：

\[
\boxed{\text{PR-AUC}}
\]

同时报告：

- ROC-AUC
- F1
- Recall
- Precision

如涉及实际风险预警，可以额外报告：

- Recall@TopK
- Precision@TopK

---

# 15. 多 Seed 实验

Few-shot 实验必须使用多个 support seeds。

推荐：

```text
Seeds = 42–61
```

共 20 个 seed。

报告：

\[
Mean \pm Std
\]

例如：

```text
PR-AUC = 0.421 ± 0.018
```

---

# 16. 推荐第一轮实验矩阵

第一轮不需要把所有实验全部跑完。

优先跑：

| Experiment | Text | Graph | Motif | OCS | LLM |
|---|---|---|---|---|---|
| Text-only LLM | ✓ | ✗ | ✗ | ✗ | ✓ |
| GNN | ✓ | ✓ | ✗ | ✗ | ✗ |
| Random + LLM | ✓ | ✓ | ✓ | ✗ | ✓ |
| GraspLLM | ✓ | ✓ | ✓ | ✓ | ✓ |

Shot：

```text
1 / 5 / 10 / 16
```

Metric：

```text
PR-AUC
ROC-AUC
F1
Recall
```

Seeds：

```text
20
```

---

# 17. 第一阶段推荐固定配置

为了尽量测试原始 GraspLLM：

```text
Graph:
    Company-only homogeneous graph

Node:
    Listed company-year

Node Text:
    Raw MDA

Node Feature:
    Frozen text embedding

Edge:
    Direct company-company relation

Edge Type:
    Removed / collapsed

Direction:
    Undirected

Motif:
    Original GraspLLM

Structure Learner:
    Original GraspLLM

OCS:
    Original GraspLLM

Projector:
    2-layer MLP

LLM:
    Frozen

Task:
    Fraud / Normal node classification
```

---

# 18. 推荐开发顺序

## Step 1

完成 FiGraph 数据读取：

```text
MDA
Company ID
Year
Label
Edges
```

---

## Step 2

生成 Company-only graph。

检查：

- 节点数量；
- fraud 数量；
- edge 数量；
- isolated nodes；
- degree distribution。

---

## Step 3

计算 motif statistics：

```text
Edge
Triangle
4-cycle
4-clique
```

确认图不会出现 motif 爆炸。

---

## Step 4

生成 MDA embeddings。

---

## Step 5

直接运行原 GraspLLM Stage-1。

优先检查是否能在不修改 structure learner 的情况下运行。

---

## Step 6

训练 projector。

LLM 与 Graph Encoder 冻结。

---

## Step 7

运行：

```text
1-shot
5-shot
10-shot
16-shot
```

---

## Step 8

加入 baselines 和 ablations。

---

# 19. 最终推荐主框架

\[
\boxed{
\begin{aligned}
&FiGraph\\
&\downarrow\\
&\text{Company-Year Nodes}\\
&+\text{Raw MDA Text}\\
&+\text{Direct Company Relations}\\
&\downarrow\\
&\text{Homogeneous Financial TAG}\\
&\downarrow\\
&\text{Frozen Text Encoder}\\
&\downarrow\\
&\text{MotifGNN}\\
&\downarrow\\
&\text{OCS}\\
&\downarrow\\
&\text{GraphToken}\\
&\downarrow\\
&\text{Projector}\\
&\downarrow\\
&\text{Frozen LLM}\\
&\downarrow\\
&\text{Fraud / Normal}
\end{aligned}
}
\]

---

# 20. 核心研究问题

最终实验应主要回答以下三个问题：

### RQ1

GraspLLM 能否从通用 TAG 迁移到金融 TAG，并有效识别财务欺诈公司？

### RQ2

相比只使用 MDA 文本：

\[
\text{MDA + Financial Graph}
\]

是否能够显著提高 few-shot fraud detection？

### RQ3

GraspLLM 的：

\[
\text{Motif Structure Learning}
+
\text{OCS}
\]

是否真正能够发现与财务欺诈相关的公司关系模式？

如果这三个问题都得到正向结果，就能够比较清晰地证明：

\[
\boxed{
\text{LLM semantic reasoning}
+
\text{financial graph structure}
}
\]

在少样本金融欺诈检测中的互补价值。
