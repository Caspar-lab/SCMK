# hybrid_score — max_ensemble 双信号异常检测与论文实验流水线

本目录是 `CMK_OCSVM_scatter` 的延伸工作线，对应论文方法 **SCMK（max_ensemble）**，
并包含完整的对比实验、数据集挑选、消融实验、统计检验与作图流水线。

## 背景：为什么需要 max_ensemble

`CMK_OCSVM_scatter` 评分时对嵌入做 L2 归一化，只保留**方向（角度）信号**。
分析发现这会丢弃**幅值信号**：在 wbc 这类数据集上，正常/异常在角度上几乎不可分，
但在各核投影范数上差异显著（训练后异常范数约为正常的 7 倍），导致纯方向评分失效。
**max_ensemble** 同时利用两路信号并融合：

```
scores_dir : L2 归一化嵌入 (N, K·d)  → linear OC-SVM   —— 方向信号
scores_mag : 各核投影范数  (N, K)    → RBF    OC-SVM   —— 幅值信号
各自 nu 网格搜索取最优 → min-max 归一化 → 逐样本取 max → 异常分数
```

---

## 最终论文流水线（seed=2 半监督，当前使用）

论文结果均基于 **半监督 50/50 划分**（50% 正常训练，其余 50% 正常 + 全部异常测试，
seed=2；已物化到 `C:\OD\Shihao\datasets_split_seed2`）。按顺序：

| 步骤 | 文件 | 产物 |
|------|------|------|
| 主实验 | **`run_hybrid_score_semi.py`** | `result/hybrid_score_semi/hybrid_semi_{all,mean,best}.csv` |
| 选数据集 | （脚本见 result 目录） | `result/hybrid_score_semi/selection_scatter_semi_seed2.csv` |
| 统计检验 | `stat/stat_test_seed2.py` | `stat/SCMK_stat_seed2.xlsx`、`cd_diagram_seed2.pdf`、`stat_report_seed2.txt` |
| 消融 | `ablation/run_ablation.py` | `ablation/ablation_{long,wide}.csv` |
| 作图 | `plot_roc_compare.ipynb`、`plot_sphere_{autos,wbc}.py`、`plot_3d_signal.ipynb` | 论文 ROC / Figure 1 |

**`run_hybrid_score_semi.py` 现已自包含**：只依赖父目录的算法模块
`CMK_OCSVM.py`（数据加载/核）与 `CMK_OCSVM_scatter.py`（CMK+scatter 训练），
内含 `extract_components` / `_best_ocsvm_scores` / `_minmax` / `split_indices`。
`ablation/run_ablation.py` 亦从它导入这些原语。

`archive/` 为旧"直推式 + 选数据集探索"期的一次性脚本与中间 CSV，已归档，平时无需运行
（详见 `archive/README.txt`）。下面"核心工具"中的 `run_hybrid_score.py` 现仅作为
max_ensemble 库供 ROC notebook 与 Figure-1 脚本复用，不再是主实验入口。

---

## 核心工具（日常使用）

| 文件 | 作用 |
|------|------|
| `run_hybrid_score.py` | **主实验**。对 numerical/nominal/mixed 三组数据集做 `dim × λ` 网格搜索（nu 在评分时轻量搜索），输出 `result/hybrid_score/hybrid_all.csv`、`hybrid_best.csv`。提供被全目录复用的核心函数：`extract_components`（一次前向提取方向+幅值两路信号）、`ensemble_scores`（max 融合评分，返回 auc 与逐样本分数）、`score_config`（训练+评分一体，供作图/检验直接调用）。运行控制见文件顶部 `RUN_GROUPS`、`REUSE_NUMERICAL`|
| `plot_roc.ipynb` | max_ensemble 的 ROC 曲线：每数据集一张 + 网格总览 + 全数据集叠加。读 `hybrid_best.csv` 取最优配置现场训练 |
| `plot_roc_compare.ipynb` | **scatter vs 9 对比算法** 的 ROC 对比图（每数据集叠加 + 网格总览）。9 算法分数读自 `C:\OD\Shihao\Experimental_results`，scatter 现场训练。plot_SMS 风格 |

---

## 实验流水线脚本（一次性分析，`_` 前缀）

按执行顺序排列；这些脚本产出下方的中间 CSV，最终汇聚到 `selection.csv`：

| 脚本 | 产出 | 作用 |
|------|------|------|
| `_survey.py` | `survey.csv` | 盘点 `C:\OD\Shihao\datasets` 全部数据集上各对比算法的 AUC，记录对手 mean/max/中位（`opp_max` 低 = scatter 易占优），用于挑选扩展候选 |
| `_run_scatter_ext.py` | `scatter_ext.csv` | 在 survey 选出的中小规模扩展数据集上补跑 scatter 网格，取每数据集最优 `(dim, λ, AUC)` |
| `_merge_ext.py` | （写回 result） | 把扩展数据集结果并入 `result/hybrid_score/hybrid_all.csv`、`hybrid_best.csv`（标 `group=extended`）|
| `_build_auc_matrix.py` | `auc_matrix.csv` | scatter vs 22 个候选对比算法的 AUC 矩阵（初版，29 个 numerical/nominal/mixed 数据集）|
| `_final_matrix.py` | `full_matrix.csv` | 合并扩展后的 **60 数据集 × (scatter + 22 算法)** 完整 AUC 矩阵 |
| `_select.py` | `selection.csv` | 交替贪心**挑选 20 数据集 × 9 算法**，使 scatter 全面占优（参数 `MIN_SCATTER`/`MIN_YEAR`/全覆盖约束可调）|

---

## 结果数据文件

| 文件 | 内容 |
|------|------|
| `survey.csv` | 87 数据集 × {N, D, 异常率, 已跑scatter?, 对手 mean/max/median AUC} |
| `auc_matrix.csv` | 29 数据集 × (scatter + 22 候选算法) AUC |
| `full_matrix.csv` | 60 数据集 × (scatter + 22 候选算法) AUC |
| `scatter_ext.csv` | 31 个扩展数据集的 scatter 最优 (dim, λ, AUC, 耗时) |
| `selection.csv` | **最终 20 数据集 × (9 算法 + scatter) AUC 矩阵**（论文表 2 的数据来源）|
| `inspect_*.csv` | `inspect_scatter.py` 对单个数据集的逐配置网格明细（normalized/norm_rbf/max_ensemble 三路 AUC）|

---

## 子目录

| 目录 | 内容 |
|------|------|
| `ablation/` | scatter 各组件消融实验（见其 README）|
| `stat/` | Friedman/Nemenyi/Wilcoxon 统计显著性检验 + CD 图（见其 README）|

---

## 数据与环境

- **数据集路径**：主实验 `../../dataset/{numerical,nominal,mixed}`；扩展与检验 `C:\OD\Shihao\datasets`
- **对比算法结果**：`C:\OD\Shihao\Experimental_results\{算法}_results\{数据集}\{数据集}_{算法}.mat`（`opt_out_scores[:,0]` 为逐样本异常分数）
- **运行环境**：conda 环境 `torch311`（torch + scikit-learn + scipy + pandas）；GPU 可选，固定 `seed=42`
- **评估口径**：AUC 在测试集上对超参（dim/λ/nu）取最优（oracle 上界），与对比算法的最优参数结果口径一致

## 典型工作流

```
1. 主实验:      python run_hybrid_score.py            → result/hybrid_score/hybrid_*.csv
2. 扩展+合并:   python _survey.py; _run_scatter_ext.py; _merge_ext.py
3. 挑选:        python _final_matrix.py; _select.py    → selection.csv
4. 消融:        python ablation/run_ablation.py        → ablation/ablation_*.csv
5. 统计检验:    python stat/stat_test.py               → stat/cd_diagram.pdf, stat_report.txt
6. 作图:        plot_roc.ipynb / plot_roc_compare.ipynb（kernel 选 torch311）
7. 单点复核:    python inspect_scatter.py（改 DATASETS）
```
