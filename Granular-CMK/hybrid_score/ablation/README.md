# ablation — scatter(max_ensemble) 组件消融实验

逐个移除/替换 SCMK 的核心组件，量化各部分对 AUC 的贡献。

## 文件

| 文件 | 作用 |
|------|------|
| `run_ablation.py` | 消融主脚本。在 8 个代表性数据集（刻意覆盖方向主导与幅值主导两类）上跑 6 个变体，输出三个 CSV。顶部 `DATASETS`、网格、训练超参可配 |
| `ablation_long.csv` | 长表：每行一个 `(数据集, 变体)` 的最优 AUC 与 `(dim, λ)` |
| `ablation_wide.csv` | 宽表：行=数据集，列=6 个变体的 AUC |
| `ablation_summary.csv` | 各变体平均 AUC 及相对 Full 的下降量 `delta_vs_full` |

## 六个消融变体

| 变体 | 移除/替换的组件 |
|------|----------------|
| `Full` | 完整方法（多核 K=5 + 跨核对比 L_cross + scatter 损失 + max_ensemble 评分）|
| `w/o_scatter` | 去掉 scatter 损失（λ=0）|
| `w/o_cross` | 去掉跨核对比 InfoNCE（仅 scatter 损失）|
| `single_kernel` | 多核 K=5 → 单核 K=1 |
| `normalized_only` | 评分仅用方向信号（linear OC-SVM）|
| `normrbf_only` | 评分仅用幅值信号（RBF OC-SVM）|

## 高效设计

6 个变体只需 **3 个训练组**（避免重复训练）：

- **TG1**（K5, cross+scatter）一次产出 `Full` / `w/o_scatter`(λ=0 子集) / `normalized_only` / `normrbf_only`
- **TG2**（K5, 仅 scatter）→ `w/o_cross`
- **TG3**（K1, 仅 scatter）→ `single_kernel`

## 主要结论（平均贡献，详见 ablation_summary.csv）

| 组件 | 移除后平均下降 |
|------|--------------|
| scatter 损失 | **−0.174**（最关键）|
| max_ensemble 融合（vs 仅方向）| −0.066 |
| max_ensemble 融合（vs 仅幅值）| −0.058 |
| 多核 | −0.036 |
| 跨核对比 | −0.009 |

核心证据：两个单信号变体各在一类数据集上崩溃（`normalized_only` 在 WBC 跌到 0.569、
`normrbf_only` 在 Autos 跌到 0.796），唯有融合在两类上都稳健 —— 这是 max_ensemble 必要性的直接证明。

## 运行

```
python run_ablation.py        # conda 环境 torch311
```
