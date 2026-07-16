"""
ab_combined_kernel_semi.py — A/B：参考论文的「学习核矩阵」下游 vs. 现 max_ensemble
===============================================================================

【动机】
  参考实现 CMK-code_release/CMK.py 的 ConLoss.forward 里，多核学习的产物是一个
  显式的组合核矩阵：
      K_combined = (1/K) Σ_k K_k        # 各视图学习嵌入上的核，逐视图取块后平均
  论文下游直接对 K_combined 做谱分解 / 聚类。而本项目 run_hybrid_score_semi.py 的
  max_ensemble 走的是「特征路线」：把嵌入喂给 linear/rbf OC-SVM，并没有显式构造
  K_combined 再作为 precomputed 核送进 OC-SVM。

  本脚本在**同一表示**（同一训练好的 CMKNet、同一 seed 划分）上，对比三条下游打分：
    A. ensemble   : 现方法 max_ensemble（linear-OCSVM 方向 + rbf-OCSVM 幅值）
    B. comb_lin   : 显式组合核 K_combined（Linear 定义）→ precomputed OC-SVM
    C. comb_gauss : 显式组合核 K_combined（Gaussian 定义, 各视图中位数带宽）→ precomputed OC-SVM

  表示固定 → 唯一变量是「下游怎么用多核学习的结果」，从而干净地度量
  “忠于参考论文的核矩阵路线是否真的提升 AUC”。

【关系（为何 B 与现方法相关而非全异）】
  方向路线的 linear-OCSVM 内部 Gram = H_dir·H_dirᵀ = Σ_k K_k^linear = K·K_combined，
  与 B 仅差常数尺度 K（不改变 OC-SVM 判定 / AUC）。故 B 约等于现方法的“方向分量”，
  差别在于 B 未叠加幅值分量、且可换用 Gaussian 组合核（C）——那才是与参考默认一致的。

【协议】
  半监督 50/50，与 run_hybrid_score_semi 完全一致：训练集只含正常样本，测试集为
  其余正常 + 全部异常；核带宽 / OC-SVM 拟合只用训练集正常样本，只对测试集打分。
  每个数据集的 (latent_dim, lambda_scatter) 取 hybrid_semi_best.csv 里现方法选出的最优配置，
  以「现方法最喜欢的表示」为公平基线。默认只跑 seed=2 划分（可在 SEEDS 调整）。

【输出】result/hybrid_score_semi/ab_combined_kernel_semi.csv
  dataset, group, dim, lambda, seed,
  auc_ensemble, auc_comb_linear, auc_comb_gaussian, d_lin, d_gau
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

_hs_dir   = os.path.dirname(os.path.abspath(__file__))
_gcmk_dir = os.path.dirname(_hs_dir)
sys.path.insert(0, _gcmk_dir)
sys.path.insert(0, _hs_dir)

# 复用现方法的原语（不重复实现），保证「表示 / 划分 / ensemble 打分」完全同源
from run_hybrid_score_semi import (
    extract_components, ensemble_auc_split, split_indices,
    resolve_path, RESULT_DIR, BEST_CSV,
)
from CMK_OCSVM import load_data, gauss_med_kernels, NU_CANDIDATES, TRAIN_CFG as BASE_CFG
from CMK_OCSVM_scatter import train_cmk_scatter


# ─── 配置 ──────────────────────────────────────────────────────────────────────

SEEDS   = [2]        # 对比划分：默认 seed=2（本组固定半监督划分）；多 seed 会按数据集取均值
FOCUS   = []         # 空 = 跑 BEST_CSV 里全部数据集；填 stem 列表只跑子集（调试用）
LIMIT   = None       # 只跑前 N 个数据集（None = 不限），控制单次运行时长
RESUME  = True       # 跳过 OUT_CSV 里已完成的数据集

OUT_CSV = os.path.join(RESULT_DIR, 'ab_combined_kernel_semi.csv')


# ─── 组合核构造：K_combined = (1/K) Σ_k K_k，各视图学习嵌入上的核 ──────────────

def _median_gamma(H_sub):
    """在（训练集）视图嵌入上用中位数启发式估 Gaussian 带宽 → gamma=1/(2·med²)。"""
    n   = len(H_sub)
    m   = min(500, n)
    idx = np.random.default_rng(0).choice(n, m, replace=False)
    d   = pairwise_distances(H_sub[idx], metric='euclidean')
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    return 1.0 / (2.0 * max(med, 1e-3) ** 2)


def combined_cross_kernel(H_norm_per, train_idx, ktype):
    """
    返回组合核的「全量×训练支持」块： (N, n_train)。
      linear   : K_k = H_k · H_k[train]ᵀ
      gaussian : K_k = exp(-γ_k · ‖·-·‖²)，γ_k 由各视图训练嵌入中位数估计
    逐视图取核后平均（严格对应参考实现 kernels.mean(2)）。
    带宽 / 支持集只依赖训练集正常样本 → 无泄漏。
    """
    K   = len(H_norm_per)
    acc = None
    for Hk in H_norm_per:
        Hk_tr = Hk[train_idx]
        if ktype == 'linear':
            Kk = Hk @ Hk_tr.T
        elif ktype == 'gaussian':
            Kk = rbf_kernel(Hk, Hk_tr, gamma=_median_gamma(Hk_tr))
        else:
            raise ValueError(ktype)
        acc = Kk if acc is None else acc + Kk
    return acc / K


def combined_kernel_auc(H_norm_per, train_idx, test_idx, y_test, nu_list, ktype):
    """显式组合核 → precomputed OC-SVM（仅训练集正常拟合，只对测试集打分），nu 网格取最优 AUC。"""
    Kx     = combined_cross_kernel(H_norm_per, train_idx, ktype)   # (N, n_train)
    K_fit  = Kx[train_idx]                                          # (n_train, n_train)
    K_scr  = Kx[test_idx]                                           # (n_test,  n_train)
    best   = -1.0
    for nu in nu_list:
        try:
            clf    = OneClassSVM(kernel='precomputed', nu=nu).fit(K_fit)
            scores = -clf.decision_function(K_scr)                 # 越大越异常
            auc    = roc_auc_score(y_test, scores)
            best   = max(best, auc)
        except Exception:
            pass
    return best if best >= 0 else float('nan')


# ─── 单数据集 × 单划分：固定表示，三条下游打分 ────────────────────────────────

def run_one_split(X, y, dim, lam, device, seed, nu_list):
    train_idx, test_idx = split_indices(y, seed)
    X_train = X[train_idx]

    kernels = gauss_med_kernels(X_train)                  # 仅训练集估计带宽
    cfg     = {**BASE_CFG, 'lambda_scatter': lam}
    y_dummy = np.zeros(len(X_train), dtype=int)
    model   = train_cmk_scatter(X_train, y_dummy, kernels, dim, device, cfg)

    H_norm_per, H_norms = extract_components(model, X, device)     # 同一表示，喂给三条路线
    y_test = (y[test_idx] != 0).astype(int)

    auc_ens = ensemble_auc_split(H_norm_per, H_norms, train_idx, test_idx, y_test, nu_list)
    auc_lin = combined_kernel_auc(H_norm_per, train_idx, test_idx, y_test, nu_list, 'linear')
    auc_gau = combined_kernel_auc(H_norm_per, train_idx, test_idx, y_test, nu_list, 'gaussian')
    return auc_ens, auc_lin, auc_gau


# ─── 主程序 ────────────────────────────────────────────────────────────────────

def load_configs():
    """从 hybrid_semi_best.csv 读每个数据集现方法选出的最优 (dim, lambda) 作为固定表示配置。"""
    df = pd.read_csv(BEST_CSV)
    cfgs = []
    for _, r in df.iterrows():
        stem = r['dataset']
        if FOCUS and stem not in FOCUS:
            continue
        path = resolve_path(stem)
        if path is None:
            continue
        cfgs.append(dict(dataset=stem, group=r['group'],
                         dim=int(r['best_dim']), lam=float(r['best_lambda']), path=path))
    if LIMIT is not None:
        cfgs = cfgs[:LIMIT]
    return cfgs


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfgs   = load_configs()

    done_rows, done = [], set()
    if RESUME and os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        done_rows, done = prev.to_dict('records'), set(prev['dataset'].unique())

    print(f'device={device}  协议=半监督 50/50  seeds={SEEDS}  nu={NU_CANDIDATES}')
    print(f'数据集={len(cfgs)}（表示固定为 hybrid_semi_best 最优配置）'
          f'{"，RESUME 跳过 "+str(len(done)) if done else ""}')
    print('=' * 92)

    rows = list(done_rows)
    todo = [c for c in cfgs if c['dataset'] not in done]
    for j, c in enumerate(todo, 1):
        stem, group, dim, lam, path = c['dataset'], c['group'], c['dim'], c['lam'], c['path']
        X, y, meta = load_data(path)

        t0 = time.time()
        ens_l, lin_l, gau_l = [], [], []
        for seed in SEEDS:
            a_e, a_l, a_g = run_one_split(X, y, dim, lam, device, seed, NU_CANDIDATES)
            ens_l.append(a_e); lin_l.append(a_l); gau_l.append(a_g)
        a_e, a_l, a_g = np.nanmean(ens_l), np.nanmean(lin_l), np.nanmean(gau_l)

        rows.append(dict(
            dataset=stem, group=group, dim=dim, lambda_scatter=lam,
            seed=(SEEDS[0] if len(SEEDS) == 1 else -1),
            auc_ensemble=round(a_e, 6),
            auc_comb_linear=round(a_l, 6),
            auc_comb_gaussian=round(a_g, 6),
            d_lin=round(a_l - a_e, 6),
            d_gau=round(a_g - a_e, 6),
        ))
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)   # 增量落盘

        print(f'[{j:>2}/{len(todo)}] {stem:<34} dim={dim:>3} lam={lam:>6}  '
              f'ens={a_e:.4f}  lin={a_l:.4f}({a_l-a_e:+.4f})  '
              f'gau={a_g:.4f}({a_g-a_e:+.4f})  ({time.time()-t0:.1f}s)')

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    print('\n' + '=' * 92)
    print('A/B 汇总（同表示，下游打分对比；Δ 为相对 max_ensemble）')
    me = df[['auc_ensemble', 'auc_comb_linear', 'auc_comb_gaussian']].mean()
    print(f'  平均 AUC   ensemble={me["auc_ensemble"]:.4f}   '
          f'comb_linear={me["auc_comb_linear"]:.4f}   comb_gaussian={me["auc_comb_gaussian"]:.4f}')
    print(f'  平均 Δ     comb_linear={df["d_lin"].mean():+.4f}   comb_gaussian={df["d_gau"].mean():+.4f}')
    n = len(df)
    print(f'  胜出计数   comb_linear>ens: {int((df["d_lin"]>0).sum())}/{n}   '
          f'comb_gaussian>ens: {int((df["d_gau"]>0).sum())}/{n}')
    best_route = max([('ensemble', me['auc_ensemble']),
                      ('comb_linear', me['auc_comb_linear']),
                      ('comb_gaussian', me['auc_comb_gaussian'])], key=lambda t: t[1])
    print(f'  → 平均最优下游路线: {best_route[0]}  (AUC={best_route[1]:.4f})')
    print(f'\n明细: {OUT_CSV}')
