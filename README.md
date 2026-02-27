# Multifaceted fused-CNN 乳腺癌 WSI 评分复现（CABCDS, 交接版）

## 1. 项目简介 (Introduction)

本项目复现了医疗 AI 论文《Multifaceted fused-CNN based scoring of breast cancer whole-slide histopathology images》（原论文无开源代码），并针对工程落地场景（极端类别不平衡、防数据泄漏、可复现训练/推理流水线等）进行了系统化改造与实现。最终在目标评估设置下达到 **Quadratic Weighted Kappa = 0.15**。

## 2. 环境与依赖 (Requirements)

本项目使用 `uv` 管理依赖，建议统一用 `uv run ...` 执行命令。

- OS：Linux（推荐 Ubuntu 22.04）
- Python：`<填写你的 Python 版本>`
- PyTorch：`<填写你的 PyTorch 版本>`
- GPU/NPU：可选
	- NVIDIA：CUDA 版本 `<填写>`
	- Ascend：`torch_npu`（如使用 NPU 推理/训练）
- 关键系统依赖
	- OpenSlide（读取 `.svs` WSI）：需安装系统库与 Python 包（确保 `openslide` 可 import）

快速检查：

```bash
uv sync
uv run python -m cabcds.roi_selector --help
uv run python -m cabcds.mf_cnn --help
uv run python -m cabcds.wsi_scorer --help
```

## 3. 数据准备 (Data Preparation)

### 3.1 TUPAC16 数据集目录约定

将 TUPAC16（WSI）和辅助标注数据按如下结构放置：

```text
dataset/
	tupac16/
		train/                          # 训练集 WSI（.svs）
		test/                           # 测试集 WSI（.svs）
		auxiliary_dataset_roi/          # ROI 标注（每个 WSI 一个 *-ROI.csv）
		auxiliary_dataset_mitoses/      # 有丝分裂辅助数据（zips / CSV GT 等，供 CNN_seg/CNN_det 使用）
	mitos14/                         # 可选：外部 mitosis 数据集（若启用 CNN_seg paper 配方）
		train/
		test/
```

说明：

- ROI CSV 文件命名要求：`<WSI_ID>-ROI.csv`（例如 `TUPAC-TR-001-ROI.csv`）。
- 产物目录默认写到 `output/`（被 `.gitignore` 忽略，适合大文件训练/推理产物）。

### 3.2 运行 ROI-Selector 生成训练数据/推理补丁

ROI-Selector（Stage 1）负责从 WSI 中抽取候选 ROI patch，供后续 CNN_global 与 Hybrid-Descriptor 使用。

```bash
# 生成正/负 patch（训练用）并更新索引 CSV
uv run python -m cabcds.roi_selector --prepare

# 仅生成正样本
uv run python -m cabcds.roi_selector --prepare-positive

# 仅生成负样本（含可选的负样本过滤）
uv run python -m cabcds.roi_selector --prepare-negative
```

推理（全片扫描 + top-k ROI 输出）走项目入口：

```bash
uv run python main.py
```

输出（关键）：

- `output/roi_selector/outputs/reports/stage_two_roi_selection.csv`
- `output/roi_selector/outputs/patches/{train|test}/<WSI_ID>/roi_*.png`

### 3.3 40x → 10x 坐标换算逻辑（交接必读）

TUPAC WSI 通常以 40x 作为 level0（基准坐标系）。ROI-Selector 在采样时会按目标倍镜（默认 10x）读取 patch：

- 设 `base_mag = 40`，`target_mag = 10`，则下采样倍数 `target_downsample = base_mag / target_mag = 4`。
- 从 OpenSlide 选择最接近该下采样倍数的读取层级 `read_level`。
- ROI CSV 中的 `(x, y, w, h)` 坐标保持 **level0 坐标系**，读取时由 OpenSlide 在内部完成 `level` 与 `downsample` 的映射；同时 patch 的“源尺寸”会按倍镜换算成 `src_patch_size = infer_patch_size * target_downsample`，保证在 10x 语义下 patch 覆盖一致区域。

这部分逻辑对应实现：`cabcds/roi_selector/utils/create_roi_patches.py::compute_sampling_params()`。

## 4. 模型训练流水线 (Training Pipeline)

本项目整体对应论文的多阶段结构（建议按顺序执行）：

1) Stage 1：ROI-Selector（传统 ML）抽取 ROI patch
2) Stage 2：MF-CNN（三个 CNN：`CNN_seg` → `CNN_det` → `CNN_global`）
3) Stage 4：Hybrid-Descriptor（每张 WSI 15 维特征）
4) Stage 5：WSI scorer（SVM 分类器输出最终分级）

下面重点写 Stage 2 的训练顺序与关键工程逻辑。

### Step 1: CNN_seg 训练（mitosis candidate segmentation）

目的：从辅助有丝分裂数据中训练一个像素级分割网络，输出候选区域（blob），为后续 `CNN_det` 提供候选框。

关键点（论文逻辑 + 工程化改造）：

- **UID 级别切分防泄漏**：按 image UID 进行 60/20/20 切分（train/val/test），避免同一原图切成 patch 后同时出现在不同集合。
- **质心标签生成**：利用 `Blue-Ratio + Otsu` 做核/有丝分裂相关区域的粗分割，并结合 centroid GT 生成训练 mask（解决“只有质心、没有像素 mask”的监督问题）。
- **极端稀疏正样本处理**：滑窗 patch 中正像素极稀疏，单纯 shuffle 会导致模型塌陷成全背景。
	- 使用 `WeightedRandomSampler` 对“含正像素的 patch”进行过采样，使训练批次中正样本出现频率提升。
	- 结合 **加权 CrossEntropy**（可通过 `CABCDS_MFCNN_SEG_POS_WEIGHT` 调整，工程上常见有效范围 50~500）。
- **Merge Val 回炉策略**：先用 val 进行调参/早停，再将 val 合并回 train 进行 final phase，以提高有效训练数据量。

运行命令（示例：按论文设置训练，具体参数可在 `python -m cabcds.mf_cnn --help` 查看）：

```bash
# 训练 CNN_seg（建议使用 GPU/NPU；device 示例：cuda:0 / npu:0 / cpu）
uv run python -m cabcds.mf_cnn \
	--train-seg-paper \
	--device <cuda:0|npu:0|cpu> \
	--batch-size 8
```

产物（示例路径，实际以 `output/mf_cnn/CNN_seg/` 为准）：

- `output/mf_cnn/CNN_seg/models/*_best.pt`
- `output/mf_cnn/CNN_seg/models/*_last.pt`
- `output/mf_cnn/CNN_seg/metrics/*.csv`

### Step 2: CNN_det 训练（mitosis / non-mitosis classification）

目的：对 `CNN_seg` 的候选 blob 进一步分类过滤，将“疑似 mitosis”与大量噪声候选区分开。

关键点：

- **从 seg 输出裁剪 patch**：对每个 blob 的 centroid 以固定窗口裁剪（默认 `80x80`）得到候选 patch，形成可落盘的训练集（index.csv）。
- **极端类别不平衡**：在真实候选分布下，正样本率可能低至 **~0.0739**（数量级 1:10~1:100），这会导致：
	- PR 曲线 / AP 非常低（这是“真实分布难度”的体现，并非一定是代码 bug）
	- 需要采样策略、hard negative mining（HNMs）、阈值选择与报告指标的联合设计

运行命令（准备数据 → 训练）：

```bash
# 1) 用 CNN_seg 生成候选并准备 CNN_det 训练集（落盘 index.csv + patches）
uv run python -m cabcds.mf_cnn \
	--prepare-det \
	--device <cuda:0|npu:0|cpu> \
	--det-seg-checkpoint <path/to/cnn_seg_best.pt>

# 2) 训练 CNN_det（可启用 WeightedRandomSampler / HNM 等选项）
uv run python -m cabcds.mf_cnn \
	--train-det-paper \
	--device <cuda:0|npu:0|cpu>
```

产物：

- `output/mf_cnn/CNN_det/runs/<run_id>/models/*(best|last|tune)*.pt`
- `output/mf_cnn/CNN_det/runs/<run_id>/det_patches/`
- `output/mf_cnn/CNN_det/runs/<run_id>/metrics/*`

### Step 3: CNN_global 训练（3-class proliferation scoring, 3-fold CV）

目的：对宏观纹理/全局模式进行评分（3 类），作为 Hybrid-Descriptor 的第 15 维 `roi_score` 的来源。

关键点：

- **宏观纹理截取**：从 WSI 以固定窗口（论文设置：`512x512`，overlap=80）抽取 `global_patches` 并生成 index.csv。
- **3 折交叉验证（CV3）**：训练 3 个 fold 模型（fold1/2/3），推理时用 **Sum Strategy** 融合（详见第 5 节）。
- **防泄漏**：划分应以 slide/group 为单位（不要把同一张 WSI 的 patch 同时分到 train/val）。

运行命令（准备数据 → 训练）：

```bash
# 1) 生成 CNN_global 训练 patch（global_patches + index.csv）
uv run python -m cabcds.mf_cnn \
	--prepare-global \
	--device <cuda:0|npu:0|cpu>

# 2) 训练 CNN_global（论文对齐：slide-level CV，设置 3 折输出 fold1/2/3）
uv run python -m cabcds.mf_cnn \
	--train-global-paper \
	--device <cuda:0|npu:0|cpu> \
	--global-paper-cv-folds 3

# （可选）由于测试集没有ground_truth,如果你的评估设置是从原始训练集中取400作为训练，100作为测试，可显式指定：
   --global-paper-holdout-train-slides 400 --global-paper-holdout-test-slides 100
```

产物：

- `output/mf_cnn/CNN_global/runs/<run_id>/models/cnn_global_paper_fold{1,2,3}.pt`
- `output/mf_cnn/CNN_global/runs/<run_id>/metrics/*`
- `output/mf_cnn/CNN_global/runs/<run_id>/global_patches/`

## 5. 特征提取与评分 (Inference & Scoring)

### 5.1 生成 15 维 Hybrid-Descriptor（Stage 4）

Hybrid-Descriptor 的每一维含义见：`cabcds/hybrid_descriptor/descriptor.py::HYBRID_DESCRIPTOR_FEATURE_NAMES`。

流程概览：

- 从 ROI-Selector 生成的 ROI patch 中，逐 patch 计算：
	- blob 统计（基于 `CNN_seg` mask + 连通域面积阈值）
	- mitosis 计数（`CNN_seg` blob → `CNN_det` 过滤）
	- ROI score（`CNN_global` 3 类得分）
- 对每张 WSI 的所有 ROI metrics 做聚合，得到 15 维向量。

运行命令（对 train/test 任选 split 导出 CSV）：

```bash
# 例：为 test split 生成特征（会自动从 output/mf_cnn 下探测 checkpoint）
uv run python scripts/extract_tupac_features.py \
	--split test \
	--out-dir output/features \
	--device <cuda:0|npu:0|cpu>

# 输出：output/features/stage_four_hybrid_descriptors.csv
```

注意：该脚本默认对 `CNN_global` 使用 **Sum Strategy** 融合 3 个 fold（`roi_scoring_ensemble_strategy = sum`），对应论文的“多折融合更稳健”的工程实现。

### 5.2 训练/推理最终 SVM（Stage 5）

最终 WSI 评分器是一个 group-keyed 的 SVM（带 StandardScaler），用于输出 3 类分级，并以 Quadratic Weighted Kappa 作为核心指标。

训练命令：

```bash
uv run python scripts/train_wsi_scorer.py \
	--features-csv <path/to/stage_four_hybrid_descriptors.csv> \
	--labels-csv <path/to/labels.csv> \
	--out-dir output/wsi_scorer
```

推理命令（模块入口，支持覆盖特征/模型/报告目录）：

```bash
uv run python -m cabcds.wsi_scorer \
	--mode predict \
	--predict-descriptor-csv <path/to/stage_four_hybrid_descriptors.csv> \
	--model-output-path <path/to/wsi_svm.joblib> \
	--report-dir <path/to/report_dir>
```

关于 fold 融合：

- `CNN_global` 在 Stage 4 通过 **Sum Strategy** 融合 3 个 fold 的预测，得到更稳定的 ROI score。
- SVM 本身是单模型（可用脚本内的小网格搜索选择 `C` / `class_weight`）。

## 6. 开发者避坑指南 (Developer Notes / Gotchas)

1) **Test AP 很低并不一定是 bug**
	 - 在 `CNN_det` 场景，候选生成后真实分布的正样本率可能低至 ~0.07，AP/PR 非常敏感。
	 - 建议同时看：阈值下的 recall、误检分布、以及最终 Stage 5 的 Kappa（端到端指标）。

2) **防泄漏是第一优先级**
	 - `CNN_seg` 必须 UID/image-level 切分；`CNN_global` 与 SVM 必须 group/slide-level 切分。
	 - 任何“按 patch 随机打散”的 split 都会造成过高的离线指标但线上失效。

3) **ROI-Selector 的负样本质量决定下游上限**
	 - 真实 WSI 中“暗色/污渍/折痕/空白”会大量进入负样本池，导致训练分布漂移。
	 - 本项目为此引入了：
		 - DL negative filter（可选，用于过滤可疑负样本）
		 - 白底过滤、暗色过滤、tissue 统计等启发式（见 `cabcds/roi_selector/utils/features.py` 等）

4) **极端不平衡必须“采样 + loss + 评估”三件套一起设计**
	 - 仅改 loss（例如加权 CE）或仅过采样都可能不稳定。
	 - 建议保留 `det_patches/` 与 `global_patches/`，后续调参/复现/重训才不会卡在数据准备环节。

5) **OpenSlide/倍率/坐标问题**
	 - ROI CSV 的坐标系、OpenSlide level 的 downsample、以及 patch resize 的语义必须一致。
	 - 优先验证：同一 ROI 坐标在 40x 与 10x 下抽到的组织区域是否一致
