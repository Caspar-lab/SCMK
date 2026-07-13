"""
stat_test_newmethods.py — SCMK vs 7 对比算法（新方法集）的统计显著性检验
========================================================================

新方法集（去掉 GBRAD/WFRDA/ILGNI/MFIOD/ECOD，换入 LMKAD/ICL/NeuTraLAD）：
  KFGOD, DisentAD, DFNO, DIF, LMKAD, ICL, NeuTraLAD  vs  SCMK
多 seed 方法(SCMK/LMKAD/ICL/NeuTraLAD)用 3-seed(0,1,2) 的 per-dataset 均值进入秩检验；
单值方法(KFGOD/DisentAD/DFNO/DIF)直接用其单一 AUC。

输入: result/hybrid_score_semi/master_compare_7algo.csv  （{算法}_mean 列）
输出(本目录 stat/，_newmethods 后缀):
  SCMK_stat_newmethods.xlsx, cd_diagram_newmethods.pdf(+.png), stat_report_newmethods.txt
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

ROOT = r'C:/OD/Shihao/5'
MASTER = os.path.join(ROOT, 'result', 'hybrid_score_semi', 'master_compare_v2.csv')
OUT = os.path.dirname(os.path.abspath(__file__))

ALGS = ['KFGOD', 'Disent', 'DeepSVDD', 'DFNO', 'LMKAD', 'ICL', 'NeuTraLAD']   # SCMK 放最后
DISP = {'Disent': 'DisentAD'}

Q05 = {2:1.960, 3:2.344, 4:2.569, 5:2.728, 6:2.850, 7:2.949, 8:3.031,
       9:3.102, 10:3.164, 11:3.219, 12:3.268}
Q10 = {2:1.645, 3:2.052, 4:2.291, 5:2.459, 6:2.589, 7:2.693, 8:2.780,
       9:2.855, 10:2.920, 11:2.978, 12:3.030}

C_AXIS, C_ALG, C_CD = 'black', 'blue', 'red'


def graph_ranks(avranks, names, cd, width=8.0, textspace=2.0, filename=None):
    avranks = list(avranks); k = len(avranks); lowv, highv = 1, k
    scalew = width - 2 * textspace
    def rankpos(r): return textspace + (r - lowv) / (highv - lowv) * scalew
    order = sorted(range(k), key=lambda i: avranks[i]); half = (k + 1) // 2
    left = order[:half]; right = order[half:][::-1]
    row, label_y0 = 0.18, 0.32
    nrows = max(len(left), len(right)); y_axis = 0.0; cd_y = -0.42; clique_y = 0.06
    height = label_y0 + nrows * row + 0.30
    fig = plt.figure(figsize=(width, height + 0.8))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, width); ax.set_ylim(height, -0.75)
    ax.plot([rankpos(lowv), rankpos(highv)], [y_axis, y_axis], color=C_AXIS, lw=2.2)
    for a in range(lowv, highv + 1):
        ax.plot([rankpos(a)] * 2, [y_axis, y_axis - 0.06], color=C_AXIS, lw=1.2)
        ax.text(rankpos(a), y_axis - 0.10, str(a), ha='center', va='bottom', color=C_AXIS, fontsize=11)
    def draw(items, is_left):
        for j, oi in enumerate(items):
            r = avranks[oi]; yy = label_y0 + j * row; xr = rankpos(r)
            xend = (textspace - 0.3) if is_left else (width - textspace + 0.3)
            ax.plot([xr, xr], [y_axis, yy], color=C_ALG, lw=1.0)
            ax.plot([xr, xend], [yy, yy], color=C_ALG, lw=1.0)
            ax.text(xend + (-0.1 if is_left else 0.1), yy, names[oi],
                    ha=('right' if is_left else 'left'), va='center', color=C_ALG, fontsize=11)
    draw(left, True); draw(right, False)
    x0, x1 = rankpos(lowv), rankpos(lowv + cd)
    ax.plot([x0, x1], [cd_y, cd_y], color=C_CD, lw=2.2)
    for xx in (x0, x1):
        ax.plot([xx, xx], [cd_y - 0.045, cd_y + 0.045], color=C_CD, lw=2.2)
    ax.text((x0 + x1) / 2, cd_y - 0.07, f'CD={cd:.2f}', ha='center', va='bottom', color=C_CD, fontsize=11)
    sr = sorted(avranks); yc = clique_y; done = -1
    for i in range(k):
        j = i
        while j + 1 < k and sr[j + 1] - sr[i] < cd:
            j += 1
        if j > i and j > done:
            ax.plot([rankpos(sr[i]) - 0.04, rankpos(sr[j]) + 0.04], [yc, yc],
                    color=C_CD, lw=2.6, solid_capstyle='round')
            yc += 0.055; done = j
    if filename:
        fig.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    df = pd.read_csv(MASTER)
    cols = [f'{a}_mean' for a in ALGS] + ['SCMK_mean']
    names = [DISP.get(a, a) for a in ALGS] + ['SCMK']
    A = df[cols].values.astype(float)
    N, k = A.shape

    R = np.vstack([rankdata(-A[i], method='average') for i in range(N)])
    avg_rank = R.mean(axis=0)
    chi2, p_fried = friedmanchisquare(*[A[:, j] for j in range(k)])
    Ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    cd05 = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))
    cd10 = Q10[k] * np.sqrt(k * (k + 1) / (6.0 * N))

    scmk = A[:, -1]
    pw = []
    for j, nm in enumerate(names[:-1]):
        try:
            _, p = wilcoxon(scmk, A[:, j], zero_method='wilcox', alternative='greater')
        except ValueError:
            p = 1.0
        pw.append((nm, A[:, j].mean(), avg_rank[j], int((scmk - A[:, j] > 0).sum()), p))
    order = np.argsort([x[4] for x in pw]); m = len(pw)
    holm = {idx: min(1.0, pw[idx][4] * (m - r)) for r, idx in enumerate(order)}

    pd.DataFrame(A, columns=names).to_excel(os.path.join(OUT, 'SCMK_stat_newmethods.xlsx'), index=False)
    graph_ranks(avg_rank, names, cd05, filename=os.path.join(OUT, 'cd_diagram_newmethods.pdf'))
    graph_ranks(avg_rank, names, cd05, filename=os.path.join(OUT, 'cd_diagram_newmethods.png'))

    L = []
    L.append(f'统计检验（新方法集, 3-seed 均值）：SCMK vs {k-1} 个对比算法，N={N}, k={k}')
    L.append('=' * 66)
    L.append(f'Friedman: chi^2={chi2:.3f}, p={p_fried:.3e} | Iman-Davenport F={Ff:.3f} (df1={k-1}, df2={(k-1)*(N-1)})')
    L.append(f'Nemenyi CD(0.05)={cd05:.3f}  CD(0.10)={cd10:.3f}')
    L.append('\n平均秩 (越小越好):')
    for nm, r in sorted(zip(names, avg_rank), key=lambda t: t[1]):
        L.append(f'  {nm:<10} {r:.3f}')
    best = avg_rank[-1]
    L.append(f'\nSCMK 平均秩 = {best:.3f}（{names[int(np.argmin(avg_rank))]} 为最优秩）')
    L.append('与 SCMK 秩差 > CD(0.05) 即显著劣于 SCMK:')
    for j, nm in enumerate(names[:-1]):
        d = avg_rank[j] - best
        L.append(f'  {nm:<10} Δrank={d:.3f}  {"[显著]" if d > cd05 else "未达显著"}')
    L.append('\n成对 Wilcoxon (SCMK>算法,单侧)+Holm:')
    L.append(f'  {"算法":<10}{"AUC均值":>9}{"avg_rank":>10}{"SCMK胜":>8}{"p":>12}{"p_holm":>12}')
    for j, (nm, mauc, ar, win, p) in enumerate(pw):
        ph = holm[j]
        mk = '***' if ph < 0.001 else ('**' if ph < 0.01 else ('*' if ph < 0.05 else 'ns'))
        L.append(f'  {nm:<10}{mauc:>9.3f}{ar:>10.3f}{win:>6}/{N}{p:>12.2e}{ph:>12.2e}  {mk}')
    L.append(f'\nSCMK: AUC均值={scmk.mean():.3f}  avg_rank={best:.3f}')
    rep = '\n'.join(L)
    print(rep)
    open(os.path.join(OUT, 'stat_report_newmethods.txt'), 'w', encoding='utf-8').write(rep + '\n')
    print(f'\n输出: cd_diagram_newmethods.pdf/.png, SCMK_stat_newmethods.xlsx, stat_report_newmethods.txt')
