# result/hybrid_score — max_ensemble 结果输出

`Granular-CMK/hybrid_score/` 流水线的结果落地目录。

## 数据文件

| 文件 | 内容 |
|------|------|
| `hybrid_best.csv` | 每数据集最优结果：`dataset, group, best_dim, best_lambda, best_auc`。共 60 个数据集（numerical 14 + nominal 6 + mixed 9 + extended 31）|
| `hybrid_all.csv` | 网格明细：`dataset, group, latent_dim, lambda_scatter, auc, train_s`。numerical/nominal/mixed 每数据集 30 个配置（dim×λ）；extended 每数据集仅最优配置 1 行 |

> `group` 取值：`numerical` / `nominal` / `mixed`（主实验三组）、`extended`（从 `C:\OD\Shihao\datasets` 补充的扩展数据集）。

## 图像文件

| 文件 / 目录 | 来源 | 内容 |
|------------|------|------|
| `roc_grid.png` | `plot_roc.ipynb` | max_ensemble 每数据集 ROC 子图网格 |
| `roc_overlay.png` | `plot_roc.ipynb` | 全数据集 ROC 叠加对比 |
| `roc_compare_grid.png` | `plot_roc_compare.ipynb` | scatter vs 9 对比算法的 ROC 网格总览 |
| `roc_compare/` | `plot_roc_compare.ipynb` | 每个数据集一张「scatter vs 9 算法」ROC 对比图（`{数据集}_ROC.png`）|

## 说明

- AUC 为测试集上对超参取最优的 oracle 上界，与对比算法口径一致
- 这些文件由 `Granular-CMK/hybrid_score/` 下的脚本/notebook 写入；重跑脚本会覆盖
- 论文表格/图：表 2 来自 `selection.csv`，CD 图来自 `stat/cd_diagram.pdf`，均在 hybrid_score 目录
