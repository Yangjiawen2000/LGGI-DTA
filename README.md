# LGGI-DTA: LLM-Guided Graph Interaction for Drug–Target Affinity Prediction

[![GitHub](https://img.shields.io/badge/GitHub-LGGI--DTA-blue?logo=github)](https://github.com/Yangjiawen2000/LGGI-DTA)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red?logo=pytorch)](https://pytorch.org/)

LGGI-DTA 是一个基于“机制级融合（Mechanism-level Fusion）”理念开发的药物-靶点亲和力（DTA）预测模型。它通过引入预训练蛋白质大语言模型（Protein LLM, 如 ESM-2），在图神经网络（GNN）的底层消息传递（Message Passing）过程中实现结构信息与语义信息的深度交互。

---

## 🌟 核心创新点

1.  **LLM-Guided Graph Interaction (LGGI) 机制**：利用蛋白质大模型的语义编码作为门控信号（Gating Signal），动态调制药物分子图中原子间的信息流动。
2.  **原子-残基级多头交互 (Multi-head Cross-Attention)**：【最新更新】引入 4 头 Cross-Attention 机制。不同注意力头可并行捕捉蛋白质残基序列中不同尺度的空间特征，让每个原子能够更精准地“按需关注”结合口袋的关键区域。
3.  **深度机制融合**：将交互过程从“预测头之前的特征拼接”提前到“特征提取过程中的动态干预”，显著提升了模型对复杂结合模式的建模能力。
4.  **原生硬件优化与兼容性**：完美适配 Apple M系列芯片 (MPS)，支持 Linux/GPU 环境。已优化适配 PyTorch 2.6+ 数据加载安全协议。

---

## 🛠 环境配置

建议使用 Conda 创建环境：

```bash
# 创建环境
conda create -n lggi_dta python=3.10
conda activate lggi_dta

# 安装必要的库
conda install -y -c conda-forge rdkit
pip install -r requirements.txt
```

对于 AutoDL RTX5090 服务器，确保 PyTorch 支持 CUDA 11.8+。

---

## 🚀 启动流程

### 1. 数据集构建
系统支持 Davis 和 KIBA 数据集。首先需要使用 ESM 模型提取蛋白质的残基级特征并构建分子图。

```bash
# 如果在国内环境，建议设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 运行数据构建脚本
python create_data.py
```
*该脚本会自动下载 ESM-2 (8M) 模型（轻量级，适合本地运行），并生成 `.pt` 文件存储在 `data/processed/` 目录下。*

### 2. 标准训练
指定数据集索引启动训练（0 为 Davis，1 为 KIBA）。

```bash
# 训练 Davis 数据集上的 LGGI-DTA 模型
python training.py 0
```

### 3. 消融实验 (Ablation Study)
我们提供了一个自动化的实验脚本 `training_ablation.py`，用于对比不同架构的性能。

支持的模型类型 (`--model`):
- `lggi_dta`: 完整模型（含多头 Cross-Attention）。
- `gnn_only`: 仅使用分子图信息（不含蛋白特征）。
- `gnn_llm_concat`: 传统的后期特征拼接方法。

运行示例：
```bash
# 在 Davis 数据集上运行消融对比
python training_ablation.py 0 --model lggi_dta --epochs 1000
```
*运行结束后，脚本会自动读取所有已完成实验的结果，并打印一份整洁的 **Performance Comparison Table**。*

---

## 📂 文件结构说明

- `models/`
    - `lggi_dta.py`: 主模型架构（集成了多头注意力机制）。
    - `gnn_only.py` / `gnn_llm_concat.py`: 对照模型实现。
- `lggi_layer.py`: 核心 LGGI 层实现（多头 Cross-Attention 逻辑）。
- `utils.py`: 数据加载库、ESM 特征提取（含缓存）及指标计算。
- `training_ablation.py`: 【新增】自动化消融实验与结果汇总脚本。
- `training.py`: 标准训练入口。
- `create_data.py`: 数据预处理流水线。

---

## 📊 评估指标
模型在每次训练后会自动计算并保存以下 5 个关键指标：
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **Pearson** (Correlation Coefficient)
- **Spearman** (Correlation Coefficient)
- **CI** (Concordance Index)

---

## 📬 引用与联系
如果您在研究中使用了本项目，请查阅相关论文或在 GitHub 提交 Issue 进行讨论。

**项目负责人**: Yangjiawen2000
**Repository**: [LGGI-DTA](https://github.com/Yangjiawen2000/LGGI-DTA)
