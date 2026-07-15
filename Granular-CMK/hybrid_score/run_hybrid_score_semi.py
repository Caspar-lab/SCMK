# 方法总体流程
# 多尺度核函数提供相似度度量
# → CMKNet 学习多核视图表示
# → cross-kernel loss 保证跨视图一致
# → scatter loss 保证正常样本紧凑
# → 提取方向信号和幅值信号
# → linear OC-SVM + RBF OC-SVM 分别打分
# → 分数融合得到最终异常分数

# 训练阶段用 CMKNet 学一个跨核一致且正常类紧凑的表示空间；
# 检测阶段从这个空间拆出方向和幅值两种信号，分别输入标准 OC-SVM 包得到异常分数，再融合。


"""
run_hybrid_score_semi.py — 半监督 50/50 划分下的混合嵌入评分（max_ensemble）
==============================================================================

【与 run_hybrid_score.py 的唯一区别：训练集 / 测试集划分协议】

  run_hybrid_score.py（旧·直推式 transductive）:
    - 全部正常样本既用于 CMK 训练、又用于 OC-SVM 拟合、还出现在评测集中
    - AUC 在「全部正常 + 全部异常」上计算 → 训练样本泄漏进测试集，AUC 偏乐观

  本脚本（新·半监督 semi-supervised，常见顶会 / 顶刊协议）:
    - 50% 正常样本 → 训练集：CMK 训练 + 高斯核带宽估计 + OC-SVM 拟合
    - 其余 50% 正常样本 + 全部异常样本 → 测试集：仅在此集合上计算 AUC
    - 训练集与测试集严格不相交，核带宽也只由训练集正常样本估计，杜绝泄漏

  为消除单次随机划分的偶然性，对每个 (latent_dim, lambda) 配置在 N_SPLITS 个
  不同随机种子的 50/50 划分上各跑一次，汇报 AUC 的均值 ± 标准差（顶会常规做法）。

【评分方式（与旧脚本一致，便于对比）】
  max_ensemble：方向信号(L2 嵌入 + linear OC-SVM) 与 幅值信号(各核范数 + RBF OC-SVM)
  各自 nu 网格搜索取最优 → 在测试集上 min-max 归一化 → 逐样本取 max → 计算测试 AUC。
  唯一改动是 OC-SVM 只在「训练集正常样本」上拟合，只对「测试集」打分。

【数据集】
  复用 result/hybrid_score/hybrid_all.csv 中跑过的全部数据集（含 group 标签），
  逐个在候选目录里定位对应 .mat：
    dataset/numerical, dataset/nominal, dataset/mixed, C:\\OD\\Shihao\\datasets

【输出 schema】（写入 result/hybrid_score_semi/）
  hybrid_semi_all.csv  : dataset, group, latent_dim, lambda_scatter, split_seed, auc, train_s
  hybrid_semi_mean.csv : dataset, group, latent_dim, lambda_scatter, auc_mean, auc_std, n_split
  hybrid_semi_best.csv : dataset, group, best_dim, best_lambda, best_auc_mean, best_auc_std
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score

_hs_dir   = os.path.dirname(os.path.abspath(__file__))
_gcmk_dir = os.path.dirname(_hs_dir)
_root_dir = os.path.dirname(_gcmk_dir)
sys.path.insert(0, _gcmk_dir)

# 仅依赖真正的算法模块（父目录）：数据加载/核 + CMK+scatter 训练
from CMK_OCSVM import (load_data, gauss_med_kernels,
                       NU_CANDIDATES, TRAIN_CFG as _BASE_CFG)
from CMK_OCSVM_scatter import train_cmk_scatter


# ─── 嵌入提取与评分原语（自包含，原 run_hybrid_score 同源）──────────────────────

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


# ─── 路径配置 ──────────────────────────────────────────────────────────────────

HYBRID_DIR  = os.path.join(_root_dir, 'result', 'hybrid_score')
SRC_ALL_CSV = os.path.join(HYBRID_DIR, 'hybrid_all.csv')   # 旧结果：取数据集 / group 清单

RESULT_DIR  = os.path.join(_root_dir, 'result', 'hybrid_score_semi')
os.makedirs(RESULT_DIR, exist_ok=True)

ALL_CSV  = os.path.join(RESULT_DIR, 'hybrid_semi_all.csv')
MEAN_CSV = os.path.join(RESULT_DIR, 'hybrid_semi_mean.csv')
BEST_CSV = os.path.join(RESULT_DIR, 'hybrid_semi_best.csv')

# 数据集 .mat 候选搜索目录（按顺序匹配，命中即止）
SEARCH_DIRS = [
    os.path.join(_root_dir, 'dataset', 'numerical'),
    os.path.join(_root_dir, 'dataset', 'nominal'),
    os.path.join(_root_dir, 'dataset', 'mixed'),
    r'C:\OD\Shihao\datasets',
]

# ─── 搜索空间与划分协议 ────────────────────────────────────────────────────────

LATENT_DIMS = [16, 32, 64, 128, 256]
LAMBDAS     = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]

TRAIN_FRAC  = 0.5                  # 训练集占正常样本比例（顶会标准 50%）
SPLIT_SEEDS = [0, 1, 2, 3, 4]      # 多次随机划分 → 报告 mean ± std

FOCUS_DATASETS = []   # 空 = 全部；填入 stem 名称只跑子集（调试 / 续跑用）
RESUME         = True # True 时跳过已写入 ALL_CSV 的数据集（支持中断续跑）


# ─── 数据集定位 ────────────────────────────────────────────────────────────────

def resolve_path(stem):
    """在候选目录中定位 stem.mat，返回首个命中路径，找不到返回 None。"""
    for d in SEARCH_DIRS:
        p = os.path.join(d, stem + '.mat')
        if os.path.exists(p):
            return p
    return None


def load_dataset_catalog():
    """从旧 hybrid_all.csv 读出 (dataset, group) 清单并解析 .mat 路径。"""
    df = pd.read_csv(SRC_ALL_CSV)
    pairs = df[['dataset', 'group']].drop_duplicates().values.tolist()
    catalog, missing = [], []
    for stem, group in pairs:
        if FOCUS_DATASETS and stem not in FOCUS_DATASETS:
            continue
        path = resolve_path(stem)
        if path is None:
            missing.append(stem)
        else:
            catalog.append(dict(dataset=stem, group=group, path=path))
    return catalog, missing


# ─── 半监督 50/50 划分 ─────────────────────────────────────────────────────────

def split_indices(y, seed, frac=TRAIN_FRAC):
    """
    将正常样本(y==0)按 frac 随机划分：
      train_idx       : frac 比例的正常样本（仅正常）
      test_idx        : 其余正常样本 + 全部异常样本
    返回 (train_idx, test_idx)，均为原始样本下标数组。
    """
    rng    = np.random.default_rng(seed)
    normal = np.where(y == 0)[0]
    anom   = np.where(y != 0)[0]
    perm   = rng.permutation(normal)
    k      = int(round(len(perm) * frac))
    train_idx = perm[:k]
    test_idx  = np.concatenate([perm[k:], anom])
    return train_idx, test_idx


# ─── 单次划分：训练 + max_ensemble 评分（测试集 AUC）────────────────────────────

def ensemble_auc_split(H_norm_per, H_norms, train_idx, test_idx, y_test, nu_list):
    """
    max_ensemble，但 OC-SVM 仅在训练集正常嵌入上拟合，只对测试集打分。

      scores_dir : 方向信号 — L2 归一化拼接嵌入 + linear OC-SVM
      scores_nrm : 幅值信号 — 各核原始范数 + RBF OC-SVM
    各自 nu 网格搜索取最优（按测试 AUC）→ 测试集上 min-max → 逐样本取 max。
    """
    H_dir = np.concatenate(H_norm_per, axis=1)            # (N, K*d)

    # _best_ocsvm_scores(H_all, y, H_normal, kernel, nu_list)
    # 这里 H_all = 测试集嵌入，H_normal = 训练集正常嵌入，y = 测试集标签
    _, _, s_dir = _best_ocsvm_scores(H_dir[test_idx],   y_test,
                                     H_dir[train_idx],   'linear', nu_list)
    _, _, s_nrm = _best_ocsvm_scores(H_norms[test_idx], y_test,
                                     H_norms[train_idx], 'rbf',    nu_list)

    if s_dir is None and s_nrm is None:
        return float('nan')
    if s_dir is None:
        combined = _minmax(s_nrm)
    elif s_nrm is None:
        combined = _minmax(s_dir)
    else:
        combined = np.maximum(_minmax(s_dir), _minmax(s_nrm))
    return roc_auc_score(y_test, combined)


def score_config_split(X, y, dim, lam, device, seed, nu_list=None):
    """
    在第 seed 个 50/50 划分上训练一个 (dim, lambda) 配置并返回测试 AUC。

    关键：核带宽与 CMK 训练只用训练集正常样本；评测只在测试集上。
    复用 train_cmk_scatter：传入 X_train + 全零标签 → 内部 y==0 过滤取全部训练样本。
    """
    if nu_list is None:
        nu_list = NU_CANDIDATES

    train_idx, test_idx = split_indices(y, seed)
    X_train = X[train_idx]

    kernels = gauss_med_kernels(X_train)                  # 仅训练集估计带宽
    cfg     = {**_BASE_CFG, 'lambda_scatter': lam}
    y_dummy = np.zeros(len(X_train), dtype=int)           # 全部视为正常 → 全用于训练
    model   = train_cmk_scatter(X_train, y_dummy, kernels, dim, device, cfg)

    H_norm_per, H_norms = extract_components(model, X, device)   # 全量一次前向
    y_test = (y[test_idx] != 0).astype(int)
    return ensemble_auc_split(H_norm_per, H_norms, train_idx, test_idx, y_test, nu_list)


# ─── 单数据集：dim × lambda 网格 × 多划分 ──────────────────────────────────────

def run_one(path, group, device):
    """对单数据集跑完整网格，每个 (dim, lambda) 在 SPLIT_SEEDS 上各跑一次。"""
    stem       = os.path.splitext(os.path.basename(path))[0]
    X, y, meta = load_data(path)
    n_dim, n_lam = len(LATENT_DIMS), len(LAMBDAS)

    rows = []
    for i_dim, dim in enumerate(LATENT_DIMS):
        for i_lam, lam in enumerate(LAMBDAS):
            aucs = []
            t0   = time.time()
            for seed in SPLIT_SEEDS:
                auc = score_config_split(X, y, dim, lam, device, seed)
                aucs.append(auc)
                rows.append(dict(
                    dataset        = stem,
                    group          = group,
                    latent_dim     = dim,
                    lambda_scatter = lam,
                    split_seed     = seed,
                    auc            = round(auc, 6),
                    train_s        = None,            # 单划分耗时不单列，见聚合耗时
                ))
            elapsed = time.time() - t0
            # 补回该配置的总耗时（均摊到每个划分行，便于事后分析）
            for r in rows[-len(SPLIT_SEEDS):]:
                r['train_s'] = round(elapsed / len(SPLIT_SEEDS), 2)

            a   = np.array(aucs, dtype=float)
            tag = '(base)' if lam == 0 else ''
            prog = f'[{i_dim+1}/{n_dim}][{i_lam+1}/{n_lam}]'
            print(f'  {prog} dim={dim:>3d}  lam={lam:>7.1f} {tag:<7}  '
                  f'AUC={np.nanmean(a):.4f}±{np.nanstd(a):.4f}  ({elapsed:.1f}s)')

    return rows


# ─── 聚合：均值表 与 最优表 ────────────────────────────────────────────────────

def mean_from_rows(all_rows):
    """对每个 (dataset, group, dim, lambda) 在划分维度上取 mean / std。"""
    df = pd.DataFrame([r for r in all_rows if not pd.isna(r['auc'])])
    if df.empty:
        return pd.DataFrame()
    agg = (df.groupby(['dataset', 'group', 'latent_dim', 'lambda_scatter'])['auc']
             .agg(auc_mean='mean', auc_std='std', n_split='count')
             .reset_index())
    agg['auc_mean'] = agg['auc_mean'].round(6)
    agg['auc_std']  = agg['auc_std'].fillna(0.0).round(6)
    return agg


def best_from_mean(df_mean):
    """对每个数据集按 auc_mean 取最优配置。"""
    best = []
    for stem, sub in df_mean.groupby('dataset'):
        br = sub.loc[sub['auc_mean'].idxmax()]
        best.append(dict(
            dataset       = stem,
            group         = br['group'],
            best_dim      = int(br['latent_dim']),
            best_lambda   = br['lambda_scatter'],
            best_auc_mean = round(float(br['auc_mean']), 6),
            best_auc_std  = round(float(br['auc_std']), 6),
        ))
    return pd.DataFrame(best)


def save_all(all_rows):
    """增量落盘三张表（防中断丢失）。"""
    pd.DataFrame(all_rows).to_csv(ALL_CSV, index=False)
    df_mean = mean_from_rows(all_rows)
    if not df_mean.empty:
        df_mean.to_csv(MEAN_CSV, index=False)
        best_from_mean(df_mean).to_csv(BEST_CSV, index=False)


def load_existing_rows():
    """RESUME 模式下读取已完成结果，返回 (rows, done_stems)。"""
    if not (RESUME and os.path.exists(ALL_CSV)):
        return [], set()
    df = pd.read_csv(ALL_CSV)
    return df.to_dict('records'), set(df['dataset'].unique())


# ─── 主程序 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    catalog, missing = load_dataset_catalog()

    n_per_ds = len(LATENT_DIMS) * len(LAMBDAS) * len(SPLIT_SEEDS)
    print(f'device={device}  协议=半监督 {int(TRAIN_FRAC*100)}/{100-int(TRAIN_FRAC*100)} 划分')
    print(f'dim={LATENT_DIMS}  lambda={LAMBDAS}  splits={SPLIT_SEEDS}')
    print(f'数据集={len(catalog)}  训练次数/数据集={n_per_ds}  评分=max_ensemble')
    if missing:
        print(f'⚠ 未定位到 .mat（已跳过）: {missing}')
    print('=' * 82)

    all_rows, done = load_existing_rows()
    if done:
        print(f'RESUME：跳过已完成 {len(done)} 个数据集')

    total      = len(catalog)
    todo       = [it for it in catalog if it['dataset'] not in done]
    n_todo     = len(todo)
    t_start    = time.time()
    ds_times   = []                       # 每数据集耗时 → 估算剩余时间(ETA)

    for j, item in enumerate(todo, start=1):
        stem, group, path = item['dataset'], item['group'], item['path']
        idx_all = len(done) + j           # 在全部数据集中的全局序号
        X, y, meta = load_data(path)
        print(f'\n[数据集 {idx_all}/{total}] (本次第 {j}/{n_todo})  '
              f'{group}/{stem}  N={meta["N"]}  D={X.shape[1]}  '
              f'anomaly={meta["anomaly_rate"]*100:.1f}%  normal={int((y==0).sum())}')

        t_ds = time.time()
        rows = run_one(path, group, device)
        all_rows.extend(rows)
        dt_ds = time.time() - t_ds
        ds_times.append(dt_ds)

        dfm = mean_from_rows(rows)
        br  = dfm.loc[dfm['auc_mean'].idxmax()]

        # 进度 / ETA：用已完成数据集的平均耗时估算剩余时间
        eta = (n_todo - j) * (sum(ds_times) / len(ds_times))
        print(f'  → best: dim={int(br["latent_dim"])}  lam={br["lambda_scatter"]}  '
              f'AUC={br["auc_mean"]:.4f}±{br["auc_std"]:.4f}  ({dt_ds:.1f}s)')
        print(f'  ⏳ 进度 {j}/{n_todo} ({100*j/n_todo:.1f}%)  '
              f'已用 {(time.time()-t_start)/60:.1f}min  预计剩余 {eta/60:.1f}min')

        save_all(all_rows)   # 增量保存

    # ── 全局汇总 ──────────────────────────────────────────────────────────────
    df_mean = mean_from_rows(all_rows)
    df_best = best_from_mean(df_mean).sort_values(['group', 'dataset']).reset_index(drop=True)
    df_best.to_csv(BEST_CSV, index=False)

    print(f'\n{"="*82}')
    print('全局汇总（半监督 50/50，max_ensemble 每数据集最优 AUC）')
    show = df_best.copy()
    show['AUC'] = (show['best_auc_mean'].map('{:.4f}'.format) + '±'
                   + show['best_auc_std'].map('{:.4f}'.format))
    print(show[['group', 'dataset', 'best_dim', 'best_lambda', 'AUC']].to_string(index=False))
    print(f'\n各组平均 AUC:')
    for grp, sub in df_best.groupby('group'):
        print(f'  {grp:<10} n={len(sub):<3} avg={sub["best_auc_mean"].mean():.4f}')
    print(f'  {"TOTAL":<10} n={len(df_best):<3} avg={df_best["best_auc_mean"].mean():.4f}')
    print(f'\n明细: {ALL_CSV}')
    print(f'均值: {MEAN_CSV}')
    print(f'汇总: {BEST_CSV}')
