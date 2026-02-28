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

将 TUPAC16 与 MITOS14 数据集按以下结构放置：

```text
dataset/
    tupac16/
        train/                          # 训练集 WSI (.svs)
        test/                           # 测试集 WSI (.svs)
        auxiliary_dataset_roi/          # ROI 标注 (*-ROI.csv)
        auxiliary_dataset_mitoses/      # 有丝分裂标注
    mitos14/                         # (可选) 外部有丝分裂数据集，用于强化
        train/                          
        test/                           
```

### 2.3 运行 ROI-Selector (Stage 1)

实现思路与策略：
- 样本定义：
    - 正样本（Positive）：来源于 TUPAC16 辅助标注，从约 500 张训练 WSI 中裁剪出 ~6809 张 positive patch，典型为组织边缘（Periphery）或高细胞密度区域。
    - 负样本（Negative）：通过人工筛选与自动生成结合，初始随机采样 ~600 张放入 `negative_raw`，再用 LabelStudio 标注 good/bad 并训练轻量二分类 CNN 进行过滤；最终生成 `negative_generated`（约 2000 张）用于训练。
- 过滤规则：剔除深色无效区域（像素值 <30 占比 >40%）等明显无效负样本；保留典型噪声（背景、大面积空白、伪影、病理笔记迹）以提升下游鲁棒性。
- Benchmark：保留独立评估集（positive 200, negative 200），不参与训练/采样。
- 坐标一致性：使用 OpenSlide 的坐标换算保证 40x(level0) → 10x 下采样语义一致，公式示例：roi_size_level0 = 5657 × (base_mag / target_mag)。
- 特征与效果：最优特征组合为 R+H+S+V 通道 + LBP + 细胞计数；在 paper-aligned 配置下，10x F-measure≈0.912, 20x≈0.955。

```bash
# 生成正/负 patch（训练用）
uv run python -m cabcds.roi_selector --prepare

# 全流程一键运行（主入口）
uv run python main.py
```

## 3. 训练流水线 (Training Pipeline)

### Step 1: CNN_seg (有丝分裂候选区域分割)

实现思路与策略：
- 任务：从 WSI 定位潜在有丝分裂候选区域（blobs），输出二值掩码供后续 `CNN_det` 使用（官方只给质心，需弱监督生成 mask）。
- 架构：基于 VGG-16 的 FCN（VGG-VD16-FCN），使用 1×1 卷积替换全连接并加入 SKIP1/2/3 跳跃连接以保留空间细节。
- 弱监督标签生成：先用 Blue-Ratio + Otsu 提取核类 Blob；若质心落在 Blob 或其 30px 半径内，将该 Blob 标为正；若未落入任一 Blob，用小圆盘围绕质心作为兜底正样本。
- 不平衡处理：使用 `WeightedRandomSampler`，正类权重示例设置为 100（显著过采样正样本以避免模型塌陷）；支持 MergeVal（val 回炉）策略在 final phase 合并训练/验证数据。
- 数据增强与合并：训练集包含 TUPAC auxiliary mitoses 与可选 MITOS14，以提升泛化性；paper-aligned 的 patch 为 512×512（overlap=80）。

评估/产出（paper-aligned）：Test GT 覆盖率 ~78.8%；典型 val 指标：dice≈0.225, iou≈0.127, recall≈0.671, acc≈0.994。

```bash
# 华为昇腾 NPU 示例: --device npu:0
# NVIDIA GPU 示例: --device cuda:0
uv run python -m cabcds.mf_cnn --train-seg-paper --device npu:0 --batch-size 8
```

### Step 2: CNN_det (有丝分裂检测过滤)

实现思路与策略：
- 任务：对 `CNN_seg` 产出的 blob（以 80×80 固定窗口裁剪）做高分辨率二分类，过滤假阳性。
- 架构：AlexNet 风格的局部二分类器（paper 实现），损失采用 Tversky/可选 focal；训练集约 12,668 patches（train=8236, val=737, test=3528），正负样本极不平衡（pos≈609，pos_rate≈0.0739）。
- 采样与 HNM：支持 WeightedRandomSampler 与 Hard Negative Mining（启用 `--det-paper-hnm`）以提升负样本难样本的权重。
- 性能说明：在自然候选分布下 AP 很低（paper 中 test AP≈0.02），主要原因是正样本极其稀疏、候选覆盖未必完整（missed GT 拉低 recall），以及 AlexNet 对微小目标表现受限。

```bash
# 1) 准备检测数据集 (支持 npu/cuda/cpu)
uv run python -m cabcds.mf_cnn --prepare-det --device npu:0 --det-seg-checkpoint output/mf_cnn/CNN_seg/models/best.pt

# 2) 训练检测模型
uv run python -m cabcds.mf_cnn --train-det-paper --device npu:0
```

### Step 3: CNN_global (全局纹理评分)

实现思路与策略：
- 任务：从每张 WSI 抽取多张 512×512 宏观纹理 patch（paper 取 top-k/slide-level patches），弱监督将 slide 标签传播给 patch 用于训练，从而得到用于 Hybrid-Descriptor 的 ROI score（第 15 维）。
- 训练策略：按 slide/group 做 3 折交叉验证（paper 对齐），可指定 holdout（例如 400 train / 100 test）；训练时可并行 fold（`--global-paper-parallel-folds`）并分配设备（`--global-paper-fold-devices`）。
- Fold 融合：采用 Sum Strategy（或 mean）对 3 折预测 softmax 做融合，得到每个 ROI 的最终 score。
- 产出示例（paper）：fold 测试 acc 大约 0.42–0.47，mean≈0.453。

```bash
# 1) 准备全局 patch
uv run python -m cabcds.mf_cnn --prepare-global --device npu:0

# 2) 3 折交叉验证训练
uv run python -m cabcds.mf_cnn --train-global-paper --device npu:0 --global-paper-cv-folds 3
```

## 4. 特征提取与最终评价 (Scoring)

### Step 4: Hybrid-Descriptor 特征提取
实现思路与策略：
- 描述符定义：对每张 WSI 的 top-4 ROI 计算 15 维混合特征，其中：
    - Facet I（特征1-5）：基于 `CNN_seg` 的 blob 统计（avg blob area, max/min/avg/no. of blobs, SD）。
    - Facet III（特征6-12）：基于 `CNN_det` 的有丝分裂计数（max/min/avg/SD 及 BRscore 转换）。
    - 衍生特征（13-14）：mitoses 与 blobs 比值（avg 与 max）。
    - Facet II（特征15）：`CNN_global` 的 ROI score（对 3 fold 预测用 SumStrategy 融合后取均值）。
- BR 转换规则（示例）：count in [0,7] → Score1；[8,14] → Score2；>14 → Score3（基于 4 个 ROI × 10 HPF 的映射逻辑）。

```bash
# 提取特征时同样可以指定 npu:0
uv run python scripts/extract_tupac_features.py --split test --device npu:0
```

### Step 5: WSI Scorer (SVM)
实现思路与策略：
- 分类器：线性 SVM（one-vs-one / `decision_function_shape=ovo`），配合 `StandardScaler` 对输入 15-D 描述符做预处理。
- 超参搜索与实验结论：
    - 通过 grid-search 比较 C∈{0.001,0.01,0.1,1,10,100} 与 class_weight∈{none,balanced}，按 mean QWK 选择最优。最佳实验（部分尝试）为 linear + `class_weight=balanced` + C=0.01（训练 400 / holdout100 的 CV 下 mean QWK≈0.098）。
    - 发现问题：阶段二（CNN_det）产生大量虚假高计数导致特征偏移，SVM 容易退化为预测多数类（QWK→0）。采取策略包括：删除 mitosis 相关特征、或将其弱化（乘以 factor），均能提升最终 QWK（示例：drop 后 QWK≈0.136，weaken 后 QWK≈0.152）。

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
