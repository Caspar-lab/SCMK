"""
stat_test_seed2.py — scatter(SCMK, 半监督 seed=2) vs 9 对比算法的统计显著性检验
================================================================================

与 stat_test.py 完全同流程（Friedman + Iman–Davenport + Nemenyi CD + 成对
Wilcoxon/Holm），唯一区别是输入换成半监督 50/50、seed=2 协议下挑选的结果：
  输入: result/hybrid_score_semi/selection_scatter_semi_seed2.csv
        （20 数据集 × {9 对比算法 + scatter_semi_seed2} 的 AUC）

  说明: scatter 列为半监督测试集 AUC；9 对比算法为全量数据 AUC（与论文表一致）。

输出（均在本目录 stat/，加 _seed2 后缀，避免覆盖原结果）：
  SCMK_stat_seed2.xlsx   —— 供 MATLAB criticaldifference2 直接跑
  cd_diagram_seed2.pdf   —— CD 图（alpha=0.05）
  stat_report_seed2.txt  —— Friedman / 平均秩 / CD / 成对 Wilcoxon 全部结论
"""

import os, sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import numpy as np
import pandas as pd
from scipy.stats import rankdata, friedmanchisquare, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 复用原脚本的 CD 图绘制，避免重复代码
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stat_test import graph_ranks, Q05, Q10, DISP

# 对比算法（本方法 SCMK 放最后）：用 DFNO 替换 LMKAD，与 selection / ROC 图一致
ALGS = ['KFGOD', 'GBRAD', 'Disent_AD', 'MFIOD', 'ILGNI', 'WFRDA', 'DFNO', 'DIF', 'ECOD']

HS   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(HS))   # .../5
SEL  = os.path.join(ROOT, 'result', 'hybrid_score_semi',
                    'selection_scatter_semi_seed2.csv')
OUT  = os.path.dirname(os.path.abspath(__file__))

SCATTER_COL = 'scatter_semi_seed2'   # 半监督 seed=2 的 scatter AUC 列
SUFFIX      = '_seed2'


# ─── 主流程 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = pd.read_csv(SEL)
    cols  = ALGS + [SCATTER_COL]
    names = [DISP.get(c, c) for c in ALGS] + ['SCMK']
    A = df[cols].values.astype(float)          # (N, k) AUC
    N, k = A.shape

    # 秩：每个数据集内 AUC 最大 -> 秩 1（method='average' 处理并列）
    R = np.vstack([rankdata(-A[i], method='average') for i in range(N)])
    avg_rank = R.mean(axis=0)

    # Friedman
    chi2, p_fried = friedmanchisquare(*[A[:, j] for j in range(k)])
    # Iman–Davenport F 修正
    Ff = (N - 1) * chi2 / (N * (k - 1) - chi2)

    cd05 = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))
    cd10 = Q10[k] * np.sqrt(k * (k + 1) / (6.0 * N))

    # 成对 Wilcoxon（SCMK vs 每个对比算法）+ Holm 校正
    scmk = A[:, -1]
    pw = []
    for j, nm in enumerate(names[:-1]):
        diff = scmk - A[:, j]
        try:
            _, p = wilcoxon(scmk, A[:, j], zero_method='wilcox',
                            alternative='greater')
        except ValueError:      # 全相等
            p = 1.0
        pw.append((nm, A[:, j].mean(), avg_rank[j], (diff > 0).sum(), p))
    # Holm 校正
    order = np.argsort([x[4] for x in pw])
    m = len(pw)
    holm = {}
    for rank_i, idx in enumerate(order):
        holm[idx] = min(1.0, pw[idx][4] * (m - rank_i))

    # ── 输出 xlsx（MATLAB criticaldifference2 兼容布局）──
    xlsx = os.path.join(OUT, f'SCMK_stat{SUFFIX}.xlsx')
    pd.DataFrame(A, columns=names).to_excel(xlsx, index=False)

    # ── CD 图 ──
    graph_ranks(avg_rank, names, cd05,
                filename=os.path.join(OUT, f'cd_diagram{SUFFIX}.pdf'))
    graph_ranks(avg_rank, names, cd05,
                filename=os.path.join(OUT, f'cd_diagram{SUFFIX}.png'))

    # ── 文本报告 ──
    lines = []
    lines.append(f'统计检验（半监督 seed=2）：SCMK vs {k-1} 个对比算法，'
                 f'N={N} 数据集，k={k} 方法')
    lines.append(f'输入: {os.path.basename(SEL)}  (scatter 列={SCATTER_COL})')
    lines.append('=' * 66)
    lines.append(f'Friedman 检验:  chi^2={chi2:.3f},  p={p_fried:.3e}')
    lines.append(f'Iman-Davenport: F={Ff:.3f}  (df1={k-1}, df2={(k-1)*(N-1)})')
    sig = '存在显著差异 (拒绝 H0)' if p_fried < 0.05 else '无显著差异'
    lines.append(f'结论: p<0.05 → {sig}')
    lines.append(f'\nNemenyi 临界差:  CD(0.05)={cd05:.3f}   CD(0.10)={cd10:.3f}')
    lines.append('\n平均秩 (越小越好):')
    for nm, r in sorted(zip(names, avg_rank), key=lambda t: t[1]):
        lines.append(f'  {nm:<10} {r:.3f}')
    best_rank = avg_rank[-1]    # SCMK
    lines.append(f'\nSCMK 平均秩 = {best_rank:.3f}'
                 f'（{names[int(np.argmin(avg_rank))]} 为最优秩）')
    lines.append('与 SCMK 平均秩之差 > CD(0.05) 即显著劣于 SCMK:')
    for j, nm in enumerate(names[:-1]):
        d = avg_rank[j] - best_rank
        mark = '[significant] 显著劣于 SCMK' if d > cd05 else '  未达显著'
        lines.append(f'  {nm:<10} Δrank={d:.3f}  {mark}')

    lines.append('\n成对 Wilcoxon signed-rank (SCMK > 算法, 单侧) + Holm 校正:')
    lines.append(f'  {"算法":<10}{"AUC均值":>9}{"avg_rank":>10}{"SCMK胜":>8}'
                 f'{"p":>12}{"p_holm":>12}{"":>4}')
    for j, (nm, mauc, ar, win, p) in enumerate(pw):
        ph = holm[j]
        mark = '***' if ph < 0.001 else ('**' if ph < 0.01 else ('*' if ph < 0.05 else 'ns'))
        lines.append(f'  {nm:<10}{mauc:>9.3f}{ar:>10.3f}{win:>6}/{N}'
                     f'{p:>12.2e}{ph:>12.2e}  {mark}')
    lines.append(f'\nSCMK: AUC均值={scmk.mean():.3f}  avg_rank={best_rank:.3f}')
    lines.append('显著性标记: *** p<0.001, ** p<0.01, * p<0.05, ns 不显著（Holm 校正后）')

    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(OUT, f'stat_report{SUFFIX}.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')

    print(f'\n输出: {xlsx}')
    print(f'      {os.path.join(OUT, f"cd_diagram{SUFFIX}.pdf")} (+.png)')
    print(f'      {os.path.join(OUT, f"stat_report{SUFFIX}.txt")}')
