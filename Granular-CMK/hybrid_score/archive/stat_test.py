"""
stat_test.py — scatter(SCMK) vs 9 对比算法的统计显著性检验
==========================================================

参考 C:\\OD\\Shihao\\stat 的 MATLAB criticaldifference2 流程（Demšar 2006）：
  Friedman 检验（多算法在多数据集上的平均秩是否存在显著差异）
  + Nemenyi 后续检验 + Critical Difference (CD) diagram。
此处用 Python 复现，并额外给出 SCMK 对各算法的成对 Wilcoxon 检验（Holm 校正）。

输入：selection.csv（20 数据集 × {9 对比算法 + scatter} 的 AUC）
输出（均在本目录 stat/）：
  SCMK_stat.xlsx   —— 与 GMKAD_results.xlsx 同布局，供 MATLAB criticaldifference2 直接跑
  cd_diagram.pdf   —— CD 图（alpha=0.05）
  stat_report.txt  —— Friedman / 平均秩 / CD / 成对 Wilcoxon 全部结论
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

HS  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEL = os.path.join(HS, 'selection.csv')
OUT = os.path.dirname(os.path.abspath(__file__))

# 对比算法（本方法 SCMK 放最后），与论文表一致
ALGS  = ['KFGOD', 'GBRAD', 'Disent_AD', 'MFIOD', 'ILGNI', 'WFRDA', 'LMKAD', 'DIF', 'ECOD']
DISP  = {'Disent_AD': 'DisentAD'}            # 展示名

# Nemenyi 两两检验临界值 q_alpha（df=inf, two-tailed），按算法数 k 索引
Q05 = {2:1.960, 3:2.344, 4:2.569, 5:2.728, 6:2.850, 7:2.949, 8:3.031,
       9:3.102, 10:3.164, 11:3.219, 12:3.268}
Q10 = {2:1.645, 3:2.052, 4:2.291, 5:2.459, 6:2.589, 7:2.693, 8:2.780,
       9:2.855, 10:2.920, 11:2.978, 12:3.030}


# ─── Demšar CD diagram（MATLAB criticaldifference 配色：轴黑/算法蓝/CD红）──────
C_AXIS, C_ALG, C_CD = 'black', 'blue', 'red'

def graph_ranks(avranks, names, cd, width=8.0, textspace=2.0, filename=None):
    """
    Critical Difference diagram，风格对齐 MATLAB criticaldifference：
      秩 1 在左、最好的算法置于左侧；主轴黑色，算法名/引线蓝色，
      CD 标尺与无显著差异连线红色；标签只显示算法名（不含秩值）。
    """
    avranks = list(avranks)
    k = len(avranks)
    lowv, highv = 1, k                       # 刻度 1..k，1 在左
    scalew = width - 2 * textspace

    def rankpos(r):
        return textspace + (r - lowv) / (highv - lowv) * scalew

    order = sorted(range(k), key=lambda i: avranks[i])   # 升序，最好在前
    half  = (k + 1) // 2
    left  = order[:half]                     # 好（秩小）→ 左侧
    right = order[half:][::-1]               # 差（秩大）→ 右侧，最差在顶

    row      = 0.18
    label_y0 = 0.32
    nrows    = max(len(left), len(right))
    y_axis   = 0.0
    cd_y     = -0.42                         # CD 标尺（轴上方）
    clique_y = 0.06                          # 无差异连线（轴下方）
    height   = label_y0 + nrows * row + 0.30

    fig = plt.figure(figsize=(width, height + 0.8))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, width)
    ax.set_ylim(height, -0.75)               # 反向：负在上、正在下

    # 主刻度轴（黑）
    ax.plot([rankpos(lowv), rankpos(highv)], [y_axis, y_axis], color=C_AXIS, lw=2.2)
    for a in range(lowv, highv + 1):
        ax.plot([rankpos(a)] * 2, [y_axis, y_axis - 0.06], color=C_AXIS, lw=1.2)
        ax.text(rankpos(a), y_axis - 0.10, str(a), ha='center', va='bottom',
                color=C_AXIS, fontsize=11)

    # 算法引线 + 名称（蓝），最好在左、最差在右
    def draw(items, is_left):
        for j, oi in enumerate(items):
            r  = avranks[oi]
            yy = label_y0 + j * row
            xr = rankpos(r)
            xend = (textspace - 0.3) if is_left else (width - textspace + 0.3)
            ax.plot([xr, xr], [y_axis, yy], color=C_ALG, lw=1.0)
            ax.plot([xr, xend], [yy, yy], color=C_ALG, lw=1.0)
            ax.text(xend + (-0.1 if is_left else 0.1), yy, names[oi],
                    ha=('right' if is_left else 'left'), va='center',
                    color=C_ALG, fontsize=11)
    draw(left, True)
    draw(right, False)

    # CD 标尺（红，轴上方左侧）
    x0, x1 = rankpos(lowv), rankpos(lowv + cd)
    ax.plot([x0, x1], [cd_y, cd_y], color=C_CD, lw=2.2)
    for xx in (x0, x1):
        ax.plot([xx, xx], [cd_y - 0.045, cd_y + 0.045], color=C_CD, lw=2.2)
    ax.text((x0 + x1) / 2, cd_y - 0.07, f'CD={cd:.2f}',
            ha='center', va='bottom', color=C_CD, fontsize=11)

    # 无显著差异组（红，轴下方；仅画极大集团）
    sr   = sorted(avranks)
    yc   = clique_y
    done = -1
    for i in range(k):
        j = i
        while j + 1 < k and sr[j + 1] - sr[i] < cd:
            j += 1
        if j > i and j > done:
            ax.plot([rankpos(sr[i]) - 0.04, rankpos(sr[j]) + 0.04], [yc, yc],
                    color=C_CD, lw=2.6, solid_capstyle='round')
            yc += 0.055
            done = j

    if filename:
        fig.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close(fig)


# ─── 主流程 ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = pd.read_csv(SEL)
    cols  = ALGS + ['scatter']
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
    xlsx = os.path.join(OUT, 'SCMK_stat.xlsx')
    pd.DataFrame(A, columns=names).to_excel(xlsx, index=False)

    # ── CD 图 ──
    graph_ranks(avg_rank, names, cd05,
                filename=os.path.join(OUT, 'cd_diagram.pdf'))
    graph_ranks(avg_rank, names, cd05,
                filename=os.path.join(OUT, 'cd_diagram.png'))

    # ── 文本报告 ──
    lines = []
    lines.append(f'统计检验：SCMK vs {k-1} 个对比算法，N={N} 数据集，k={k} 方法')
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
    lines.append(f'\nSCMK 平均秩 = {best_rank:.3f}（{names[int(np.argmin(avg_rank))]} 为最优秩）')
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
    with open(os.path.join(OUT, 'stat_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')

    print(f'\n输出: {xlsx}')
    print(f'      {os.path.join(OUT, "cd_diagram.pdf")} (+.png)')
    print(f'      {os.path.join(OUT, "stat_report.txt")}')
