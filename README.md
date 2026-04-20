# LGGI-DTA: LLM-Guided Graph Interaction for Drug–Target Affinity Prediction

[![GitHub](https://img.shields.io/badge/GitHub-LGGI--DTA-blue?logo=github)](https://github.com/Yangjiawen2000/LGGI-DTA)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red?logo=pytorch)](https://pytorch.org/)

LGGI-DTA 是一个基于“机制级融合（Mechanism-level Fusion）”理念开发的药物-靶点亲和力（DTA）预测模型。它打破了传统模型中仅在最后阶段进行特征拼接（Late Fusion）的局限，通过引入预训练蛋白质大语言模型（Protein LLM, 如 ESM-2），在图神经网络（GNN）的底层消息传递（Message Passing）过程中实现结构信息与语义信息的深度交互。

---

## 🌟 核心创新点

1.  **LLM-Guided Graph Interaction (LGGI) 机制**：利用蛋白质大模型的语义编码作为门控信号（Gating Signal），动态调制药物分子图中原子间的信息流动。
2.  **原子-残基级交互 (Atom-Residue Cross-Attention)**：通过 Cross-Attention 机制，让每一个原子能够“按需关注”蛋白质特定的残基片段，生成个性化的引导信号。
3.  **深度机制融合**：将交互过程从“预测头之前的特征拼接”提前到“特征提取过程中的动态干预”，显著提升了模型对复杂结合模式的建模能力。
4.  **原生硬件优化**：完美适配 Apple M系列芯片 (MPS)，并在底层算子层面做了 CPU 后备兼容处理，确保在 macOS 和 Linux/GPU 环境下均能稳定高效运行。

---

## 🛠 环境配置

建议使用 Conda 创建环境：

```bash
# 创建环境
conda create -n geometric2 python=3.10
conda activate geometric2

# 安装必要的库
conda install -y -c conda-forge rdkit
pip install torch torch-geometric transformers tqdm pandas numpy scipy accelerate
```

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

### 2. 模型训练
指定数据集索引启动训练（0 为 Davis，1 为 KIBA）。

```bash
# 训练 Davis 数据集上的 LGGI-DTA 模型
python training.py 0
```
*默认优先使用 `mps` (macOS) 或 `cuda` (GPU) 加速。*

### 3. 消融实验与验证
项目中已预置了多组对照模型，用于验证创新点的有效性：
- `models/gnn_only.py`: 仅使用分子的图结构信息。
- `models/gnn_llm_concat.py`: 使用传统的后期特征拼接方法。

支持 80/20 比例的训练/验证划分：
```bash
python training_validation.py 0
```

---

## 📂 文件结构说明

- `models/`
    - `lggi_dta.py`: 主模型代码库。
    - `gnn_only.py` / `gnn_llm_concat.py`: 消融实验对照模型。
- `lggi_layer.py`: 核心 LGGI 层实现（含 Cross-Attention 逻辑）。
- `utils.py`: 数据加载库、ESM 特征提取（含缓存机制）及 5 大评估指标实现。
- `create_data.py`: 数据预处理流水线。
- `training.py`: 标准训练入口。
- `training_validation.py`: 带验证集划分的优选训练入口。

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
