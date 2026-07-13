"""
run_sensitivity.py — SCMK 参数敏感性分析（隐藏层维度 d、scatter 权重 λ、OC-SVM 的 ν）
================================================================================

数据集：result/hybrid_score_semi/selection_scatter_semi_seed2.csv 中的 20 个；
种子：半监督 seed ∈ {0,1,2}，每条曲线对数据集取均值、误差带为跨数据集 std。

  d  与  λ ：直接取自 result/hybrid_score_semi/hybrid_semi_all.csv
             （其 auc 已在 ν∈{0.01,0.05,0.1,0.2} 上内部取最优）。
  ν        ：CSV 未记录，本脚本在每个数据集的最优 (d,λ) 上补跑 OC-SVM 评分阶段
             （不重训网络的话需要每数据集训练一次，再在 ν 网格上评分）。

输出（本目录）：
  sens_dim.csv / sens_lambda.csv / sens_nu.csv / heatmap_dim_lambda.csv
  fig_sens_dim.pdf(+png) / fig_sens_lambda.* / fig_sens_nu.* / fig_heatmap.*
  fig_sensitivity_all.*（三联图）
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

HS = r'C:\OD\Shihao\5\Granular-CMK\hybrid_score'
sys.path.insert(0, HS)
from run_hybrid_score_semi import (split_indices, extract_components, _minmax,
                                   gauss_med_kernels, load_data, _BASE_CFG)
from CMK_OCSVM_scatter import train_cmk_scatter
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

ROOT = r'C:/OD/Shihao/5'
HERE = os.path.dirname(os.path.abspath(__file__))
DR = 'C:/OD/Shihao/datasets'
SEEDS = [0, 1, 2]
DIMS = [16, 32, 64, 128, 256]
LAMBDAS = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
NUS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Times New Roman'

sel = pd.read_csv(ROOT + '/result/hybrid_score_semi/selection_scatter_semi_seed2.csv')
DATASETS = list(sel['dataset'])
allr = pd.read_csv(ROOT + '/result/hybrid_score_semi/hybrid_semi_all.csv')
allr = allr[allr['dataset'].isin(DATASETS) & allr['split_seed'].isin(SEEDS)]


# ───────────────── d 与 λ 敏感性（取自 CSV）─────────────────

def marginal(param, fixed_best_over):
    """对 param 的每个取值：每数据集先在另一参数上取最优、再对 seed 取均值，
    最后对数据集求 mean/std。返回 (values, mean, std)。"""
    vals = sorted(allr[param].unique())
    rows = []
    for v in vals:
        sub = allr[allr[param] == v]
        per_ds = []
        for ds in DATASETS:
            d = sub[sub['dataset'] == ds]
            # 每 seed 在 fixed_best_over 上取最优，再对 seed 平均
            best_per_seed = d.groupby('split_seed')['auc'].max()
            per_ds.append(best_per_seed.mean())
        a = np.array(per_ds, float)
        rows.append((v, a.mean(), a.std(ddof=1)))
    v, m, s = zip(*rows)
    return np.array(v), np.array(m), np.array(s)

dim_v, dim_m, dim_s = marginal('latent_dim', 'lambda_scatter')
lam_v, lam_m, lam_s = marginal('lambda_scatter', 'latent_dim')
pd.DataFrame({'latent_dim': dim_v, 'auc_mean': dim_m, 'auc_std': dim_s}).to_csv(
    os.path.join(HERE, 'sens_dim.csv'), index=False)
pd.DataFrame({'lambda': lam_v, 'auc_mean': lam_m, 'auc_std': lam_s}).to_csv(
    os.path.join(HERE, 'sens_lambda.csv'), index=False)

# d × λ 热图（对数据集与 seed 求均值）
heat = np.zeros((len(DIMS), len(LAMBDAS)))
for i, dd in enumerate(DIMS):
    for j, ll in enumerate(LAMBDAS):
        sub = allr[(allr['latent_dim'] == dd) & (allr['lambda_scatter'] == ll)]
        heat[i, j] = sub.groupby('dataset')['auc'].mean().mean()
pd.DataFrame(heat, index=DIMS, columns=LAMBDAS).to_csv(os.path.join(HERE, 'heatmap_dim_lambda.csv'))


# ───────────────── ν 敏感性（补跑 OC-SVM 评分阶段）─────────────────

# 每数据集的最优 (d,λ)：用 seed2 的最优配置（与论文一致）
s2 = allr[allr['split_seed'] == 2]
BEST_CFG = {}
for ds in DATASETS:
    sub = s2[s2['dataset'] == ds]
    br = sub.loc[sub['auc'].idxmax()]
    BEST_CFG[ds] = (int(br['latent_dim']), float(br['lambda_scatter']))


def nu_curve_for(ds, seed):
    """在最优 (d,λ) 下训练一次，对每个 ν 算 max-ensemble 测试 AUC。"""
    X, y, _ = load_data(os.path.join(DR, ds + '.mat'))
    tr, te = split_indices(y, seed)
    dim, lam = BEST_CFG[ds]
    ker = gauss_med_kernels(X[tr])
    model = train_cmk_scatter(X[tr], np.zeros(len(tr), int), ker, dim, device,
                              {**_BASE_CFG, 'lambda_scatter': lam})
    Hnp, Hn = extract_components(model, X, device)
    Hd = np.concatenate(Hnp, axis=1)
    yte = (y[te] != 0).astype(int)
    out = {}
    for nu in NUS:
        try:
            sd = -OneClassSVM(kernel='linear', nu=nu).fit(Hd[tr]).decision_function(Hd[te])
            sn = -OneClassSVM(kernel='rbf', nu=nu).fit(Hn[tr]).decision_function(Hn[te])
            out[nu] = roc_auc_score(yte, np.maximum(_minmax(sd), _minmax(sn)))
        except Exception:
            out[nu] = np.nan
    return out


if __name__ == '__main__':
    print(f'device={device}  datasets={len(DATASETS)}  seeds={SEEDS}', flush=True)
    # 收集 ν 曲线：per (dataset, seed)
    nu_records = []
    for ds in DATASETS:
        for seed in SEEDS:
            c = nu_curve_for(ds, seed)
            nu_records.append(dict(dataset=ds, seed=seed, **{f'nu_{n}': c[n] for n in NUS}))
        print(f'  {ds} done', flush=True)
    nudf = pd.DataFrame(nu_records)
    # 每数据集先对 seed 平均，再对数据集 mean/std
    nu_mean, nu_std = [], []
    for n in NUS:
        per_ds = nudf.groupby('dataset')[f'nu_{n}'].mean()
        nu_mean.append(per_ds.mean()); nu_std.append(per_ds.std(ddof=1))
    pd.DataFrame({'nu': NUS, 'auc_mean': nu_mean, 'auc_std': nu_std}).to_csv(
        os.path.join(HERE, 'sens_nu.csv'), index=False)
    nudf.to_csv(os.path.join(HERE, 'sens_nu_raw.csv'), index=False)

    # ───────────────── 作图 ─────────────────
    def lineplot(ax, x, m, s, xlabel, logx=False, xticklabels=None):
        ax.plot(x, m, '-o', color='#DC143C', lw=1.8, ms=5, zorder=3)
        ax.fill_between(x, m - s, m + s, color='#DC143C', alpha=0.18, zorder=1)
        if logx: ax.set_xscale('log')
        ax.set_xlabel(xlabel); ax.set_ylabel('Mean AUC')
        ax.grid(True, alpha=0.3)
        if xticklabels is not None:
            ax.set_xticks(x); ax.set_xticklabels(xticklabels)

    # λ 用 log，但含 0 → 用类别位置
    lam_x = np.arange(len(lam_v))
    nu_mean = np.array(nu_mean); nu_std = np.array(nu_std)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    lineplot(axes[0], np.array(dim_v, float), dim_m, dim_s, 'Latent dimension $d$',
             xticklabels=[str(int(v)) for v in dim_v])
    axes[0].set_xticks(np.array(dim_v, float))
    axes[0].set_title('(a) Hidden dimension')
    lineplot(axes[1], lam_x, lam_m, lam_s, 'Scatter weight $\\lambda$',
             xticklabels=[('0' if v == 0 else f'{v:g}') for v in lam_v])
    axes[1].set_title('(b) Scatter weight')
    lineplot(axes[2], np.array(NUS, float), nu_mean, nu_std, 'OC-SVM $\\nu$')
    axes[2].set_title('(c) OC-SVM $\\nu$')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'fig_sensitivity_all.{ext}'), bbox_inches='tight', dpi=200)
    plt.close(fig)

    # 单独热图
    fig, ax = plt.subplots(figsize=(5.2, 4))
    im = ax.imshow(heat, aspect='auto', cmap='viridis', origin='lower')
    ax.set_xticks(range(len(LAMBDAS))); ax.set_xticklabels([('0' if v == 0 else f'{v:g}') for v in LAMBDAS])
    ax.set_yticks(range(len(DIMS))); ax.set_yticklabels(DIMS)
    ax.set_xlabel('Scatter weight $\\lambda$'); ax.set_ylabel('Latent dimension $d$')
    for i in range(len(DIMS)):
        for j in range(len(LAMBDAS)):
            ax.text(j, i, f'{heat[i,j]:.3f}', ha='center', va='center',
                    color='white' if heat[i, j] < heat.max() - 0.04 else 'black', fontsize=7)
    fig.colorbar(im, ax=ax, label='Mean AUC')
    ax.set_title('Mean AUC over 20 datasets')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, f'fig_heatmap.{ext}'), bbox_inches='tight', dpi=200)
    plt.close(fig)

    print('\n=== 敏感性汇总（20 数据集, seeds 0/1/2 均值）===')
    print('d     :', dict(zip([int(v) for v in dim_v], np.round(dim_m, 4))))
    print('lambda:', dict(zip([f'{v:g}' for v in lam_v], np.round(lam_m, 4))))
    print('nu    :', dict(zip(NUS, np.round(nu_mean, 4))))
    print(f'\nd 极差={dim_m.max()-dim_m.min():.4f}  λ 极差={lam_m.max()-lam_m.min():.4f}  '
          f'ν 极差={nu_mean.max()-nu_mean.min():.4f}')
    print('图: fig_sensitivity_all.pdf, fig_heatmap.pdf  |  表: sens_*.csv, heatmap_dim_lambda.csv')
