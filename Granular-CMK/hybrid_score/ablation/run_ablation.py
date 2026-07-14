"""
run_ablation.py — scatter(max_ensemble) 消融实验
================================================

逐个移除/替换核心组件，验证各部分对 AUC 的贡献。

【6 个消融变体】
  Full             完整方法：多核(K=5) + 跨核对比 L_cross + scatter损失 + max_ensemble评分
  w/o_scatter      去掉 scatter 损失（λ=0，仅跨核对比）
  w/o_cross        去掉跨核对比 L_cross（仅 scatter 损失，无跨核一致性约束）
  single_kernel    单核(K=1)（K=1 时无核对，自然无 L_cross，仅 scatter 损失）
  normalized_only  评分仅用方向信号（L2归一化嵌入 + linear OCSVM）
  normrbf_only     评分仅用幅值信号（各核范数 + RBF OCSVM）

【高效设计：6 变体仅需 3 个训练组】
  TG1 = (K5, cross+scatter) 训练一次 → Full / w/o_scatter(λ=0) / normalized_only / normrbf_only
  TG2 = (K5, scatter only)            → w/o_cross
  TG3 = (K1, scatter only)            → single_kernel
  每个变体取其网格(dim×lambda)上的最优 AUC（与主实验 oracle 口径一致）。
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score

# 上级目录（hybrid_score）在 path 中以便复用主实验函数
_HS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HS)

# 全部复用主实验脚本 run_hybrid_score_semi（自包含）+ 父目录算法模块；不再依赖 run_hybrid_score
from run_hybrid_score_semi import (split_indices, extract_components,
                                   _best_ocsvm_scores, _minmax)
from CMK_OCSVM import load_data, gauss_med_kernels, CMKNet, cross_kernel_loss
from CMK_OCSVM_scatter import scatter_loss

# 半监督协议：与主实验 / datasets_split_seed2 完全一致
# 训练集 = 50% 正常样本；测试集 = 其余 50% 正常 + 全部异常。仅在测试集上评测 AUC。
SPLIT_SEED = 2

# ════════════════════════════════════════════════════════════════════════════
#                                参 数 配 置
# ════════════════════════════════════════════════════════════════════════════

# 消融用代表性数据集（覆盖方向主导/幅值主导/不同维度）；可增删
DATASETS = [
    'autos_variant1',              # mixed, 幅值主导, scatter 碾压
    'cardio',                      # numerical, 方向信号强
    'bands_band_6_variant1',       # mixed, scatter=1.0
    'ionosphere_b_24_variant1',    # numerical, 方向主导, =1.0
    'vowels',                      # 扩展, 高 AUC
    'ecoli',                       # 扩展
    'sonar_M_10_variant1',         # 高维, =1.0
    'wbc_malignant_39_variant1',   # 幅值主导典型(方向信号失效)
]

SEARCH_DIRS = [
    r'C:\OD\Shihao\datasets',
    r'C:\OD\Shihao\5\dataset\numerical',
    r'C:\OD\Shihao\5\dataset\nominal',
    r'C:\OD\Shihao\5\dataset\mixed',
]

LATENT_DIMS   = [16, 32, 64, 128, 256]
LAMBDAS       = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
NU_CANDIDATES = [0.01, 0.05, 0.1, 0.2]

KERNEL_RATIOS_FULL   = (0.1, 0.5, 1.0, 2.0, 5.0)   # 5 个多尺度核
KERNEL_RATIOS_SINGLE = (1.0,)                       # 单核（中位带宽）

EPOCHS, BATCH_SIZE, LR, NORMALIZE, SEED = 100, 512, 0.01, True, 42

RESULT_DIR = os.path.dirname(os.path.abspath(__file__))
LONG_CSV   = os.path.join(RESULT_DIR, 'ablation_long.csv')
WIDE_CSV   = os.path.join(RESULT_DIR, 'ablation_wide.csv')

VARIANT_ORDER = ['Full', 'w/o_scatter', 'w/o_cross', 'single_kernel',
                 'normalized_only', 'normrbf_only']

# ════════════════════════════════════════════════════════════════════════════


def locate(name):
    if name.lower().endswith('.mat') and os.path.exists(name):
        return name
    for d in SEARCH_DIRS:
        p = os.path.join(d, name + '.mat')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'找不到 {name}')


def train(Xtr, kernels, dim, device, lam, use_cross, use_scatter):
    """自定义训练，支持 L_cross / scatter 损失开关。Xtr=训练集正常样本 (N_tr, D)。"""
    torch.manual_seed(SEED); np.random.seed(SEED)
    N, D = Xtr.shape
    model = CMKNet(D, dim, len(kernels), NORMALIZE).to(device)
    opt   = optim.Adam(model.parameters(), lr=LR)
    Xt    = torch.tensor(Xtr, dtype=torch.float32)
    for _ in range(EPOCHS):
        perm = torch.randperm(N)
        for i in range(0, N, BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            if len(idx) < 4:
                continue
            hs = model(Xt[idx].to(device))
            terms = []
            if use_cross and len(kernels) >= 2:
                terms.append(cross_kernel_loss(hs, kernels))
            if use_scatter and lam > 0:
                terms.append(lam * scatter_loss(hs))
            if not terms:
                continue
            loss = sum(terms)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def eval_sources(model, X, y, tr, te, device):
    """
    半监督评测：OC-SVM 仅在训练集(tr)正常嵌入上拟合，只对测试集(te)打分，
    AUC 在测试集标签上计算。返回 (normalized_auc, norm_rbf_auc, max_ensemble_auc)。
    """
    Hn, Hnorms = extract_components(model, X, device)
    Hdir = np.concatenate(Hn, axis=1)
    yte  = (y[te] != 0).astype(int)
    ad, _, sd = _best_ocsvm_scores(Hdir[te],   yte, Hdir[tr],   'linear', NU_CANDIDATES)
    an, _, sn = _best_ocsvm_scores(Hnorms[te], yte, Hnorms[tr], 'rbf',    NU_CANDIDATES)
    if sd is None or sn is None:
        ae = max(ad, an)
    else:
        ae = roc_auc_score(yte, np.maximum(_minmax(sd), _minmax(sn)))
    return ad, an, ae


def run_dataset(name, device):
    path = locate(name)
    stem = os.path.splitext(os.path.basename(path))[0]
    X, y, meta = load_data(path)
    tr, te = split_indices(y, SPLIT_SEED)        # seed=2 划分（= datasets_split_seed2）
    Xtr_n = X[tr]                                # 训练集（仅 50% 正常样本）
    kern5 = gauss_med_kernels(Xtr_n, ratios=KERNEL_RATIOS_FULL)
    kern1 = gauss_med_kernels(Xtr_n, ratios=KERNEL_RATIOS_SINGLE)
    lam_pos = [l for l in LAMBDAS if l > 0]

    res = {v: -1.0 for v in VARIANT_ORDER}
    cfg = {v: None for v in VARIANT_ORDER}

    def upd(v, auc, dim, lam):
        if auc > res[v]:
            res[v] = auc; cfg[v] = (dim, lam)

    t0 = time.time()

    # ── TG1: K5, cross+scatter（含 λ=0）→ Full / w/o_scatter / normalized_only / normrbf_only ──
    for dim in LATENT_DIMS:
        for lam in LAMBDAS:
            m = train(Xtr_n, kern5, dim, device, lam, use_cross=True, use_scatter=True)
            ad, an, ae = eval_sources(m, X, y, tr, te, device)
            upd('Full', ae, dim, lam)
            upd('normalized_only', ad, dim, lam)
            upd('normrbf_only', an, dim, lam)
            if lam == 0.0:
                upd('w/o_scatter', ae, dim, lam)

    # ── TG2: K5, scatter only（无 cross, λ>0）→ w/o_cross ──
    for dim in LATENT_DIMS:
        for lam in lam_pos:
            m = train(Xtr_n, kern5, dim, device, lam, use_cross=False, use_scatter=True)
            _, _, ae = eval_sources(m, X, y, tr, te, device)
            upd('w/o_cross', ae, dim, lam)

    # ── TG3: K1, scatter only（单核, λ>0）→ single_kernel ──
    for dim in LATENT_DIMS:
        for lam in lam_pos:
            m = train(Xtr_n, kern1, dim, device, lam, use_cross=False, use_scatter=True)
            _, _, ae = eval_sources(m, X, y, tr, te, device)
            upd('single_kernel', ae, dim, lam)

    dt = time.time() - t0
    full = res['Full']
    print(f'\n[{stem}]  N={meta["N"]} D={X.shape[1]}  ({dt:.0f}s)')
    for v in VARIANT_ORDER:
        d = res[v] - full
        tag = '(完整)' if v == 'Full' else f'Δ={d:+.4f}'
        print(f'    {v:<16} AUC={res[v]:.4f}  {tag}  cfg={cfg[v]}')

    rows = [dict(dataset=stem, variant=v, auc=round(res[v], 4),
                 best_dim=cfg[v][0] if cfg[v] else None,
                 best_lambda=cfg[v][1] if cfg[v] else None)
            for v in VARIANT_ORDER]
    return rows


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  消融数据集={len(DATASETS)}  变体={VARIANT_ORDER}')
    print('=' * 78)

    all_rows = []
    for name in DATASETS:
        try:
            rows = run_dataset(name, device)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(LONG_CSV, index=False)
            # 宽表
            wide = pd.DataFrame(all_rows).pivot(index='dataset', columns='variant', values='auc')
            wide = wide[VARIANT_ORDER]
            wide.to_csv(WIDE_CSV)
        except Exception as e:
            print(f'[ERR] {name}: {e}')

    # ── 汇总 ──
    wide = pd.DataFrame(all_rows).pivot(index='dataset', columns='variant', values='auc')[VARIANT_ORDER]
    print(f'\n{"="*78}\n消融结果汇总 (AUC，行=数据集 列=变体):')
    print(wide.to_string())
    print(f'\n各变体平均 AUC 及相对 Full 的下降:')
    full_mean = wide['Full'].mean()
    for v in VARIANT_ORDER:
        m = wide[v].mean()
        print(f'  {v:<16} mean={m:.4f}  Δ_vs_Full={m-full_mean:+.4f}')
    print(f'\n明细: {LONG_CSV}\n宽表: {WIDE_CSV}')
