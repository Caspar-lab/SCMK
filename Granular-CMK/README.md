# Granular-CMK 项目文件总结

本项目围绕 **CMK（Contrastive Multiple Kernel）** 展开，探索其在异常检测和聚类两个方向上的改进，核心创新包括：粒球加速、跨核对比学习（CKAD）、OC-SVM 后端、散度惩罚、以及随机游走评分。

---

## 项目结构概览

```
Granular-CMK/
├── 粒球基础设施
│   ├── Granular_ball.py        # GrainBall 数据结构（早期探索）
│   └── GB.py                   # 粒球生成算法核心（getGranularBall）
│
├── 算法核心模块
│   ├── GCMK.py                 # 粒球加速 CMK 聚类
│   ├── CMK_AD.py               # CMK 混合型数据异常检测（两视图）
│   ├── CKAD.py                 # 跨核对比异常检测（核即视图）
│   ├── CMK_OCSVM.py            # CMK + OC-SVM（自适应高斯核）
│   └── CMK_OCSVM_scatter.py    # CMK + 散度惩罚 + OC-SVM
│
├── 实验运行脚本
│   ├── run_gb_generation.py    # 粒球生成可行性验证（早期探索）
│   ├── run_gcmk_bbcsport.py    # GCMK 聚类实验（BBCSport）
│   ├── run_autos.py            # CMK_AD 单数据集快速验证
│   ├── run_cmk_ad.py           # CMK_AD 混合数据集批量实验
│   ├── run_kernel_ablation.py  # 核组合消融（7 种核配置）
│   ├── run_iris_ablation.py    # iris 低维数据核消融
│   ├── run_cardio.py           # CKAD 完整消融（四种评分器）
│   ├── run_numerical_all.py    # CMK+OC-SVM 数值数据集批量
│   ├── run_nominal_all.py      # CMK+OC-SVM 标称数据集批量
│   ├── run_numerical_scatter.py# CMK+散度惩罚 数值数据集批量
│   └── run_blend_search.py     # 核混合权重 w 扫描实验
│
├── CMK-RW/                     # 随机游走改进方向
│   ├── cmk_rw.py               # CMK-RW 三阶段算法
│   └── run_numerical.py        # CMK-RW 数值数据集批量实验
│
└── hybrid_score/               # max_ensemble 双信号方法 + 论文实验流水线
    ├── run_hybrid_score.py     #   主实验（方向+幅值融合评分）
    ├── inspect_scatter.py      #   单数据集效果检验工具
    ├── plot_roc*.ipynb         #   ROC 作图（自身 / vs 对比算法）
    ├── ablation/               #   组件消融实验
    └── stat/                   #   Friedman/Nemenyi/Wilcoxon 统计检验
```

---

## 各文件详细说明

### 粒球基础设施

| 文件 | 作用 |
|------|------|
| `Granular_ball.py` | 定义 `GrainBall` 数据类（center/radius/weight/density），早期 OOP 封装，已被 `GB.py` 取代，不再被直接调用 |
| `GB.py` | 粒球生成算法核心，实现 `getGranularBall(data)` 函数。算法分两阶段：先按密度分裂（`division_ball`），再按半径归一化（`normalized_ball`），输出球心矩阵、每球样本数和索引。被 `GCMK.py` 和 `run_gb_generation.py` 调用 |

---

### 算法核心模块

#### `GCMK.py` — Granular-CMK 聚类

将粒球压缩与 CMK 对比核学习结合，用于多视图聚类。

**流程：**
1. 在全特征空间生成共享粒球划分 → 各视图球心矩阵（样本数压缩）
2. `FCNet`（各视图独立线性投影 W）在球心上训练 → 加权对比核损失（`GranularConLoss`）
3. CMK 阶段（前 1/3 epoch）：仅对比损失；CMKKM 阶段（后 2/3 epoch）：追加核 k-means 对齐损失
4. 聚类标签通过球成员关系传播回原始样本

**支持的核：** Gaussian / Linear / Polynomial / Sigmoid / Cauchy

---

#### `CMK_AD.py` — CMK 混合型数据异常检测

针对含标称列和数值列的混合型数据集，用标称/数值两个视图做 CMK 对比训练，输出多核拼接嵌入，接 KNN 异常评分。

**特点：**
- 自动检测标称列（独热编码）与数值列（MinMaxScaler）
- Mini-batch 训练，每种核独立训练一个 `FCNet`
- 对比三种方案：KNN 基线 / 单核 CMK+KNN / 多核 CMK+KNN（嵌入拼接）

---

#### `CKAD.py` — 跨核对比异常检测（核即视图）

**核心创新：** 不依赖特征划分，以不同核函数作为隐式视图学习跨核一致嵌入。

**模型：** `CKADNet`，K 个独立线性投影头（各对应一种核）
**损失：** 对所有 C(K,2) 核对 (k,l) 计算跨核 InfoNCE，同一样本在两核下的嵌入为正对
**四种评分器：**
- `CK-Incon`：跨核不一致性（||h_k - h_l||² 均值）
- `CK-KNN`：拼接嵌入 KNN 距离
- `OC-SVM`：线性核单分类 SVM（实验最优）
- `SVDD`：RBF 核 OC-SVM（最小包围超球面）

**半监督：** 仅正常样本参与训练；测试阶段对全量样本打分。

---

#### `CMK_OCSVM.py` — CMK + OC-SVM（主力方法）

在纯数值型/标称型/混合型数据上的 CMK + 线性 OC-SVM 流程，是后续改进的基础模块。

**关键设计：**
- `gauss_med_kernels`：基于正常样本欧氏距离中位数自适应生成 5 个高斯核（×[0.1, 0.5, 1.0, 2.0, 5.0]）
- `CMKNet`：K 个独立无偏置线性投影头
- `cross_kernel_loss`：跨核 InfoNCE（与 `CKAD.py` 相同机制）
- `best_nu_ocsvm`：在 [0.01, 0.05, 0.1, 0.2] 网格搜索最优 nu
- 支持 `.mat` 和 `.csv` 两种数据格式，解决 Windows 中文路径编码问题

---

#### `CMK_OCSVM_scatter.py` — CMK + 散度惩罚 + OC-SVM

在 `CMK_OCSVM.py` 基础上增加 **多核散度惩罚**（CMKKM 单类紧凑性思路）：

$$L_{total} = L_{cross} + \lambda \cdot L_{scatter}, \quad L_{scatter} = -\frac{1}{K}\sum_k \|\mu_k\|^2$$

其中 μ_k 为第 k 个投影头批次嵌入的均值中心。最大化 μ 模长 = 最小化正常样本散度，使 OC-SVM 边界更紧。

还实现了 `eval_ocsvm_blend`：检测阶段对原始特征 RBF 核与 CMK 嵌入线性核做凸组合搜索最优混合权重 w。

---

### 实验运行脚本

| 文件 | 作用 |
|------|------|
| `run_gb_generation.py` | 早期探索脚本，验证仅用正常样本生成粒球的可行性，将球心/权重/索引保存为 `.npz`，供后续阶段读取（路径已硬编码旧路径，已停用） |
| `run_gcmk_bbcsport.py` | 在 BBCSport 双视图数据集上遍历 5 种核运行 GCMK，epochs=450（与原 CMK 论文对齐），结果保存在 `save/` 目录 |
| `run_autos.py` | 用 `CMK_AD.run_experiment` 对 `autos_variant1.mat` 做单次快速验证，对比 KNN 基线/单核/多核 CMK |
| `run_cmk_ad.py` | 批量在 `dataset/mixed` 下所有 `.mat` 文件运行 `CMK_AD`，打印汇总 AUC 表格 |
| `run_kernel_ablation.py` | 使用 `CKAD` 在混合数据集上对比 7 种核配置（Hetero-5 / Gauss-5-lin / Gauss-5-log / Gauss-5-med / Gauss-2+het / Cauchy-5 / Poly-5），指标为 CK-Incon 和 CK-KNN |
| `run_iris_ablation.py` | 针对极低维（D=4）的 iris 数据集，在 5 种核组合 × 5 个 latent_dim × linear/rbf 两种 OC-SVM 核做全因子消融 |
| `run_cardio.py` | 在纯数值数据集（cardio.mat 等）上用 CKAD + Gauss-5-med 扫描 latent_dim，对比四种评分器（CK-Incon / CK-KNN / OC-SVM / SVDD），保存结果到 `result/` |
| `run_numerical_all.py` | 批量在 `dataset/numerical` 下运行 `CMK_OCSVM`，扫描 latent_dim=[16,32,64,128,256]，每数据集取最优维度写汇总 CSV |
| `run_nominal_all.py` | 同上，针对 `dataset/nominal` 下的 `.csv` 标称数据集 |
| `run_numerical_scatter.py` | 批量在 `dataset/numerical` 下运行 `CMK_OCSVM_scatter`，固定 λ_scatter，与 CMK_OCSVM 历史结果对比 |
| `run_blend_search.py` | 在数值数据集上训练一次 CMK（scatter 版），然后扫描混合权重 w∈{0, 0.25, 0.5, 0.75, 1.0}，评估 oracle-best-w 是否超过两端点 |

---

### CMK-RW 子目录

#### `CMK-RW/cmk_rw.py` — CMK-RW 三阶段算法

对 `CMK+OC-SVM` 的三项改进：

| 阶段 | 内容 |
|------|------|
| Stage 1 | 跨核对比训练（同 CMK_OCSVM），得到拼接嵌入 H_all |
| Stage 2 | 在 H_all 上计算多样化核矩阵（多尺度高斯 + 余弦核），优化**逐样本**自适应核权重 η_i（Adam，图保持损失 + 核多样性正则） |
| Stage 3 | 用 η 融合核构建全样本相似度矩阵，做**随机游走**（带重启），稳态分布 φ_i 越低则异常 |

思路源自 DMFAD，用 CMK 嵌入替换原始特征提供更有判别力的表示。

#### `CMK-RW/run_numerical.py` — CMK-RW 批量实验

批量在 `dataset/numerical` 上运行 CMK-RW，与已有 CMK+OC-SVM 结果对比，打印汇总表格。

---

## 开发顺序

```
阶段 1：粒球基础设施
  Granular_ball.py → GB.py → run_gb_generation.py

阶段 2：CMK 聚类方向（迁移自 CMK-code_release）
  GCMK.py → run_gcmk_bbcsport.py

阶段 3：CMK 混合型数据异常检测
  CMK_AD.py → run_autos.py → run_cmk_ad.py

阶段 4：跨核创新（CKAD）
  CKAD.py → run_kernel_ablation.py → run_iris_ablation.py → run_cardio.py

阶段 5：CMK + OC-SVM（数值/标称数据集主线）
  CMK_OCSVM.py → run_numerical_all.py → run_nominal_all.py

阶段 6：散度惩罚改进
  CMK_OCSVM_scatter.py → run_numerical_scatter.py → run_blend_search.py

阶段 7：随机游走改进（CMK-RW）
  CMK-RW/cmk_rw.py → CMK-RW/run_numerical.py
```

---

## 依赖关系图

```
GB.py
  └── GCMK.py ─────────────── run_gcmk_bbcsport.py

CMK_OCSVM.py ────────────────── run_numerical_all.py
  ├── CMK_OCSVM_scatter.py ──── run_numerical_scatter.py
  │     └── run_blend_search.py
  ├── CMK-RW/cmk_rw.py ──────── CMK-RW/run_numerical.py
  └── run_nominal_all.py

CMK_AD.py ───────────────────── run_autos.py
  └── run_cmk_ad.py

CKAD.py ─────────────────────── run_kernel_ablation.py
  ├── run_iris_ablation.py (也引用 CMK_OCSVM)
  └── run_cardio.py
```

---

## 保存目录说明

| 路径 | 内容 |
|------|------|
| `save/` | GCMK 聚类实验中间核矩阵 `.mat`（K, H, K_gb, H_gb），格式与 CMK 原始代码兼容 |
| `../result/` | 异常检测实验 CSV 结果（AUC 汇总、明细），由各 `run_*.py` 脚本写入 |
| `../result/hybrid_score/` | max_ensemble 结果（`hybrid_all/best.csv`、ROC 图），详见该目录 README |

---

## hybrid_score 工作线（max_ensemble，论文主线）

阶段 6 的散度惩罚（`CMK_OCSVM_scatter.py`）只用 L2 归一化的**方向信号**，会丢弃投影
**幅值信号**，在 wbc 等数据集上失效。`hybrid_score/` 子目录提出 **max_ensemble**：
方向信号（linear OC-SVM）与幅值信号（各核范数 + RBF OC-SVM）逐样本取 max 融合，
并配套完整的对比实验、数据集挑选、消融、统计检验与作图流水线，对应论文
`../CMK_OCSVM_scatter_latex/elsarticle/manuscript.tex`。

详见 **`hybrid_score/README.md`**（及其 `ablation/`、`stat/` 子目录 README）。
