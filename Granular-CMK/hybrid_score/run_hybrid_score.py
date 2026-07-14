"""
run_hybrid_score.py — 混合嵌入评分（max_ensemble），多数据集组
================================================================

【背景：两类判别结构】
  ionosphere 类：正常/异常在余弦域差异大（方向信号），幅值无差异
                 → L2 归一化嵌入 + linear OC-SVM 有效
  wbc 类        ：正常/异常原始投影幅值差异大（训练后 7.38x），余弦域几乎无差异
                 → L2 归一化丢弃幅值信号导致失效，需各核范数 + RBF OC-SVM

【max_ensemble：自适应组合两种信号】
  scores_dir : L2 归一化嵌入 (N, K*d) → linear OC-SVM     —— 方向信号
  scores_nrm : 各核原始范数 (N, K)     → RBF OC-SVM        —— 幅值信号
  各自 nu 网格搜索取最优 → min-max 归一化至 [0,1] → 逐样本取 max → 计算 AUC

【数据集组】
  dataset/numerical, dataset/nominal, dataset/mixed 三个目录。
  RUN_GROUPS 控制本次训练哪些组；REUSE_NUMERICAL 复用已有 numerical 结果
  （seed 固定，重跑结果一致，跳过大数据集省时）。三组结果统一写入 hybrid_all/best.csv。

【输出 schema】
  hybrid_all.csv : dataset, group, latent_dim, lambda_scatter, auc, train_s
  hybrid_best.csv: dataset, group, best_dim, best_lambda, best_auc
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

_gcmk_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root_dir = os.path.dirname(_gcmk_dir)
sys.path.insert(0, _gcmk_dir)

from CMK_OCSVM import (load_data, gauss_med_kernels,
                       NU_CANDIDATES, TRAIN_CFG as _BASE_CFG)
from CMK_OCSVM_scatter import train_cmk_scatter

# ─── 路径配置 ──────────────────────────────────────────────────────────────────

DATASET_ROOT = os.path.join(_root_dir, 'dataset')
DATA_DIRS = {
    'numerical': os.path.join(DATASET_ROOT, 'numerical'),
    'nominal':   os.path.join(DATASET_ROOT, 'nominal'),
    'mixed':     os.path.join(DATASET_ROOT, 'mixed'),
}
RESULT_DIR = os.path.join(_root_dir, 'result', 'hybrid_score')
os.makedirs(RESULT_DIR, exist_ok=True)

ALL_CSV  = os.path.join(RESULT_DIR, 'hybrid_all.csv')
BEST_CSV = os.path.join(RESULT_DIR, 'hybrid_best.csv')

# 兼容旧 DATA_DIR 引用（如 notebook/外部脚本）
DATA_DIR = DATA_DIRS['numerical']

# ─── 运行控制 ──────────────────────────────────────────────────────────────────

RUN_GROUPS      = ['nominal', 'mixed']   # 本次训练的组
REUSE_NUMERICAL = True                    # 复用现有 hybrid_all.csv 中的 numerical 结果

# ─── 搜索空间 ──────────────────────────────────────────────────────────────────

LATENT_DIMS = [16, 32, 64, 128, 256]
LAMBDAS     = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]

FOCUS_DATASETS = []   # 空 = 该组全部；填入 stem 名称只跑子集


# ─── 嵌入提取 ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_components(model, X, device, batch_size=2048):
    """
    一次前向传播提取两种信号：
      H_norm_per : L2 归一化嵌入列表，每核 (N, d) —— 方向/余弦信号
      H_norms    : 各核原始投影范数 (N, K)         —— 幅值信号
    """
    model.eval()
    X_t     = torch.tensor(X, dtype=torch.float32)
    per_raw = [[] for _ in model.projectors]
    for i in range(0, len(X), batch_size):
        xb     = X_t[i: i + batch_size].to(device)
        hs_raw = [p(xb) for p in model.projectors]
        for k, h in enumerate(hs_raw):
            per_raw[k].append(h.cpu().numpy())
    H_raw_per  = [np.concatenate(p, axis=0) for p in per_raw]
    H_norm_per = [h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
                  for h in H_raw_per]
    H_norms    = np.concatenate(
        [np.linalg.norm(h, axis=1, keepdims=True) for h in H_raw_per], axis=1
    )  # (N, K)
    return H_norm_per, H_norms


# ─── 评分 ─────────────────────────────────────────────────────────────────────

def _best_ocsvm_scores(H_all, y, H_normal, kernel, nu_list):
    """遍历 nu，返回 AUC 最高时的 (best_auc, best_nu, best_scores)。"""
    best_auc, best_nu, best_scores = -1.0, nu_list[0], None
    for nu in nu_list:
        try:
            clf    = OneClassSVM(kernel=kernel, nu=nu).fit(H_normal)
            scores = -clf.decision_function(H_all)   # 越大越可能是异常
            auc    = roc_auc_score(y, scores)
            if auc > best_auc:
                best_auc, best_nu, best_scores = auc, nu, scores
        except Exception:
            pass
    return best_auc, best_nu, best_scores


def _minmax(s):
    """min-max 归一化至 [0,1]，恒为常数时返回全 0。"""
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(s)


def ensemble_scores(H_norm_per, H_norms, y, nu_list=None):
    """
    max_ensemble 评分。返回 (auc, scores)。

    scores_dir : 方向信号（L2 归一化嵌入 + linear OC-SVM）
    scores_nrm : 幅值信号（各核范数 + RBF OC-SVM）
    各自取 nu 最优 → min-max 归一化 → 逐样本取 max。
    """
    if nu_list is None:
        nu_list = NU_CANDIDATES

    H_dir = np.concatenate(H_norm_per, axis=1)   # (N, K*d)
    _, _, scores_dir = _best_ocsvm_scores(H_dir,   y, H_dir[y == 0],   'linear', nu_list)
    _, _, scores_nrm = _best_ocsvm_scores(H_norms, y, H_norms[y == 0], 'rbf',    nu_list)

    if scores_dir is None and scores_nrm is None:
        return float('nan'), None
    if scores_dir is None:
        combined = _minmax(scores_nrm)
    elif scores_nrm is None:
        combined = _minmax(scores_dir)
    else:
        combined = np.maximum(_minmax(scores_dir), _minmax(scores_nrm))

    return roc_auc_score(y, combined), combined


def score_config(X, y, kernels, dim, lam, device, nu_list=None):
    """
    训练一个 (dim, lambda) 配置并返回 max_ensemble 的 (auc, scores)。
    供 ROC 绘图等下游分析直接复用。
    """
    cfg   = {**_BASE_CFG, 'lambda_scatter': lam}
    model = train_cmk_scatter(X, y, kernels, dim, device, cfg)
    H_norm_per, H_norms = extract_components(model, X, device)
    return ensemble_scores(H_norm_per, H_norms, y, nu_list)


# ─── 单数据集搜索 ──────────────────────────────────────────────────────────────

def run_one(path, group, device):
    stem       = os.path.splitext(os.path.basename(path))[0]
    X, y, meta = load_data(path)
    kernels    = gauss_med_kernels(X[y == 0])
    n_dim, n_lam = len(LATENT_DIMS), len(LAMBDAS)

    rows = []
    for i_dim, dim in enumerate(LATENT_DIMS):
        for i_lam, lam in enumerate(LAMBDAS):
            t0  = time.time()
            auc, _ = score_config(X, y, kernels, dim, lam, device)
            elapsed = time.time() - t0

            rows.append(dict(
                dataset        = stem,
                group          = group,
                latent_dim     = dim,
                lambda_scatter = lam,
                auc            = round(auc, 6),
                train_s        = round(elapsed, 2),
            ))

            prog = f'[{i_dim+1}/{n_dim}][{i_lam+1}/{n_lam}]'
            tag  = '(base)' if lam == 0 else ''
            print(f'  {prog} dim={dim:>3d}  lam={lam:>7.1f} {tag:<7}  '
                  f'max_ensemble={auc:.4f}  ({elapsed:.1f}s)')

    return rows


# ─── 复用已有 numerical 结果 ──────────────────────────────────────────────────

def load_prior_rows(group_name='numerical'):
    """
    从现有 hybrid_all.csv 提取 max_ensemble 明细，标准化为统一 schema。

    兼容旧格式（含 emb_mode/nu 列，多模式）：仅取 emb_mode=='max_ensemble'。
    旧实验仅覆盖 numerical，故缺失 group 列时一律标注为 group_name。
    """
    if not os.path.exists(ALL_CSV):
        return []
    df = pd.read_csv(ALL_CSV)
    if 'emb_mode' in df.columns:
        df = df[df['emb_mode'] == 'max_ensemble']
    if 'group' not in df.columns:
        df = df.assign(group=group_name)
    df = df[df['group'] == group_name]
    cols = ['dataset', 'group', 'latent_dim', 'lambda_scatter', 'auc']
    df = df[[c for c in cols if c in df.columns]].copy()
    if 'train_s' not in df.columns:
        df['train_s'] = np.nan
    return df.to_dict('records')


def best_from_rows(all_rows):
    """对全量明细按数据集取 AUC 最优行，生成 best 汇总。"""
    df = pd.DataFrame([r for r in all_rows if not pd.isna(r['auc'])])
    best = []
    for stem, sub in df.groupby('dataset'):
        br = sub.loc[sub['auc'].idxmax()]
        best.append(dict(
            dataset     = stem,
            group       = br['group'],
            best_dim    = int(br['latent_dim']),
            best_lambda = br['lambda_scatter'],
            best_auc    = round(float(br['auc']), 6),
        ))
    return best


# ─── 主程序 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_train = len(LATENT_DIMS) * len(LAMBDAS)
    print(f'device={device}  RUN_GROUPS={RUN_GROUPS}  REUSE_NUMERICAL={REUSE_NUMERICAL}')
    print(f'dim={LATENT_DIMS}  lambda={LAMBDAS}  mode=max_ensemble  训练次数/数据集={n_train}')
    print('=' * 82)

    all_rows = []

    # 复用现有 numerical（若启用且本次不重跑 numerical）
    if REUSE_NUMERICAL and 'numerical' not in RUN_GROUPS:
        prior = load_prior_rows('numerical')
        all_rows.extend(prior)
        n_ds = len({r['dataset'] for r in prior})
        print(f'\n复用 numerical 已有结果: {len(prior)} 行 / {n_ds} 数据集')

    # 训练 RUN_GROUPS 指定的组
    for group in RUN_GROUPS:
        data_dir = DATA_DIRS[group]
        files = sorted(f for f in os.listdir(data_dir) if f.endswith('.mat'))
        if FOCUS_DATASETS:
            files = [f for f in files if os.path.splitext(f)[0] in FOCUS_DATASETS]
        print(f'\n{"#"*82}\n# group={group}  datasets={len(files)}\n{"#"*82}')

        for fname in files:
            path = os.path.join(data_dir, fname)
            stem = os.path.splitext(fname)[0]
            X, y, meta = load_data(path)
            print(f'\n[{group}/{stem}]  N={meta["N"]}  D={X.shape[1]}  '
                  f'anomaly={meta["anomaly_rate"]*100:.1f}%  normal={int((y==0).sum())}')

            t_ds = time.time()
            rows = run_one(path, group, device)
            all_rows.extend(rows)

            df_ds = pd.DataFrame([r for r in rows if not pd.isna(r['auc'])])
            br    = df_ds.loc[df_ds['auc'].idxmax()].to_dict()
            print(f'  → best: dim={int(br["latent_dim"])}  lam={br["lambda_scatter"]}  '
                  f'AUC={br["auc"]:.4f}  ({time.time()-t_ds:.1f}s)')

            # 增量保存（防中断丢失）
            pd.DataFrame(all_rows).to_csv(ALL_CSV, index=False)
            pd.DataFrame(best_from_rows(all_rows)).to_csv(BEST_CSV, index=False)

    # 最终汇总
    df_best = pd.DataFrame(best_from_rows(all_rows))
    df_best = df_best.sort_values(['group', 'dataset']).reset_index(drop=True)
    df_best.to_csv(BEST_CSV, index=False)

    print(f'\n{"="*82}')
    print('全局汇总（max_ensemble 每数据集最优 AUC）')
    print(df_best[['group', 'dataset', 'best_dim', 'best_lambda', 'best_auc']].to_string(index=False))
    print(f'\n各组平均 AUC:')
    for grp, sub in df_best.groupby('group'):
        print(f'  {grp:<10} n={len(sub):<3} avg={sub["best_auc"].mean():.4f}')
    print(f'  {"TOTAL":<10} n={len(df_best):<3} avg={df_best["best_auc"].mean():.4f}')
    print(f'\n明细: {ALL_CSV}')
    print(f'汇总: {BEST_CSV}')
