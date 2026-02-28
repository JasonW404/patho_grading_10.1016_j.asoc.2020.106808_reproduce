# PATHO_GRADING: Multifaceted Fused-CNN for Breast Cancer WSI Scoring

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. 项目简介 (Introduction)

本项目是医疗 AI 论文 《Multifaceted fused-CNN based scoring of breast cancer whole-slide histopathology images》 (10.1016/j.asoc.2020.106808) 的高质量复现。本项目针对工程落地场景（极端类别不平衡、防数据泄漏、可复现训练/推理流水线等）进行了系统化重构。最终在目标评估设置下达到 **Quadratic Weighted Kappa = 0.15**。

**核心能力：**
- **全流程覆盖**：从 WSI (Whole Slide Image) 的 ROI 提取、有丝分裂检测到最终的癌症分级。
- **模块化设计**：包含 ROI-Selector (Stage 1), MF-CNN (Stage 2), Hybrid-Descriptor (Stage 3), 及 WSI Scorer (Stage 4)。
- **工程化优化**：支持 `uv` 依赖管理，内置防泄漏切分、加权采样等医学图像处理最佳实践。

## 2. 快速启动 (Quick Start)

### 2.1 环境准备

本项目建议使用 [uv](https://github.com/astral-sh/uv) 管理 Python 环境。

```bash
# 安装依赖
uv sync

# 验证安装
uv run python -m cabcds.roi_selector --help
uv run python -m cabcds.mf_cnn --help
uv run python -m cabcds.wsi_scorer --help
```

### 2.2 数据放置 (Data Preparation)

将 TUPAC16 数据集按以下结构放置：

```text
dataset/
    tupac16/
        train/                          # 训练集 WSI (.svs)
        test/                           # 测试集 WSI (.svs)
        auxiliary_dataset_roi/          # ROI 标注 (*-ROI.csv)
        auxiliary_dataset_mitoses/      # 有丝分裂标注
```

### 2.3 运行 ROI-Selector (Stage 1)

```bash
# 生成正/负 patch（训练用）
uv run python -m cabcds.roi_selector --prepare

# 全流程一键运行（主入口）
uv run python main.py
```

## 3. 训练流水线 (Training Pipeline)

### Step 1: CNN_seg (有丝分裂候选区域分割)
```bash
# 华为昇腾 NPU 示例: --device npu:0
# NVIDIA GPU 示例: --device cuda:0
uv run python -m cabcds.mf_cnn --train-seg-paper --device npu:0 --batch-size 8
```

### Step 2: CNN_det (有丝分裂检测过滤)
```bash
# 1) 准备检测数据集 (支持 npu/cuda/cpu)
uv run python -m cabcds.mf_cnn --prepare-det --device npu:0 --det-seg-checkpoint output/mf_cnn/CNN_seg/models/best.pt

# 2) 训练检测模型
uv run python -m cabcds.mf_cnn --train-det-paper --device npu:0
```

### Step 3: CNN_global (全局纹理评分)
```bash
# 1) 准备全局 patch
uv run python -m cabcds.mf_cnn --prepare-global --device npu:0

# 2) 3 折交叉验证训练
uv run python -m cabcds.mf_cnn --train-global-paper --device npu:0 --global-paper-cv-folds 3
```

## 4. 特征提取与最终评价 (Scoring)

### Step 4: Hybrid-Descriptor 特征提取
```bash
# 提取特征时同样可以指定 npu:0
uv run python scripts/extract_tupac_features.py --split test --device npu:0
```

### Step 5: WSI Scorer (SVM)
```bash
# 训练 SVM 分类器
uv run python scripts/train_wsi_scorer.py --features-csv output/features/train.csv --labels-csv labels.csv

# 推理预测
uv run python -m cabcds.wsi_scorer --mode predict --predict-descriptor-csv output/features/test.csv
```

## 5. 开发者避坑指南 (Developer Notes)

1. **硬件选型 (Hardware)**: 本项目兼容华为昇腾 (Ascend) NPU 与 NVIDIA CUDA。在运行命令时通过 `--device npu:0` 或 `--device cuda:0` 进行切换即可。
2. **防泄漏 (No Data Leakage)**: 任何划分必须以 slide/group 为单位，严禁按 patch 随机打散。
2. **类别不平衡**: 本项目内置了 `WeightedRandomSampler` 和 HNM (Hard Negative Mining) 策略。
3. **坐标系**: ROI 坐标基于 Level 0 (40x)，读取时会自动换算至 10x 语义空间。

## 6. 目录结构 (Repository Structure)

```text
cabcds/        # 核心逻辑 (roi_selector, mf_cnn, hybrid_descriptor, wsi_scorer)
scripts/       # 业务脚本 (特征提取、训练触发)
tools/         # 数据集构建与对比工具
output/        # 模型检查点与推理产物
```

---
*本项目采用 MIT 协议。*
