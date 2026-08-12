# GraspLLM YelpZip 欺诈检测测试

本仓库仅用于测试 **GraspLLM** 框架在 YelpZip 评论欺诈检测任务上的效果，并非 GraspLLM 官方仓库。

当前实验将 YelpZip 评论构造成两种独立的文本属性图：

- `yelpzip_rur`：使用相同用户关系连接评论节点。
- `yelpzip_rbr`：使用相同商家关系连接评论节点。

仓库保留原始 GraspLLM 的 Stage 0、Stage 1、Stage 2、Stage 3 流程，并增加 YelpZip 数据预处理、RUR/RBR 独立测试、随机基线、zero-shot 概率评估以及冻结模型的 1/5/10-shot in-context learning 测试。

数据集、模型权重、checkpoint 和实验输出不包含在本仓库中。YelpZip 的完整测试流程与命令见 [YELPZIP_BASE.md](YELPZIP_BASE.md)。

## 原始项目

- GraspLLM 官方代码：[Heinz217/GraspLLM](https://github.com/Heinz217/GraspLLM)
- GraspLLM 论文：[GraspLLM: Towards Zero-Shot Generalization on Text-Attributed Graphs with LLMs](https://arxiv.org/abs/2606.11898)
- 官方数据集：[Heinz217/GraspLLM-Datasets](https://huggingface.co/datasets/Heinz217/GraspLLM-Datasets)

本仓库中的 GraspLLM 相关代码与方法归原作者所有；这里仅提供面向 YelpZip 欺诈检测的实验适配与测试代码。
