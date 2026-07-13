archive/ —— 旧探索期代码（已归档，非当前论文流水线）
=====================================================

这些是早期"直推式评分 + 数据集挑选"阶段的一次性脚本与中间产物，
已被当前 seed=2 半监督流水线取代（见上级目录与 result/hybrid_score_semi/）。
保留仅作追溯，平时无需运行。

脚本：
  _build_auc_matrix.py / _final_matrix.py / _merge_ext.py / _run_scatter_ext.py
  _survey.py / _select.py            —— 旧的 AUC 矩阵汇总 + 数据集挑选流程
  _regen_roc.py                      —— 旧 ROC 复刻（已被 plot_roc_compare.ipynb 取代）
  inspect_scatter.py                 —— 单数据集诊断工具
  stat_test.py                       —— 旧统计检验（已被 stat/stat_test_seed2.py 取代）

中间 CSV：
  auc_matrix.csv / full_matrix.csv / survey.csv / scatter_ext.csv
  selection.csv / selection_20.csv / selection_ngbad_rod.csv
  inspect_*.csv

注意：deepsvdd/run_deepsvdd.py 曾读取 selection.csv（现已移到本目录）。
若要重跑该遗留 baseline，将其 SEL 路径指向 archive/selection.csv 即可。
当前论文的最终挑选结果在 result/hybrid_score_semi/selection_scatter_semi_seed2.csv。
