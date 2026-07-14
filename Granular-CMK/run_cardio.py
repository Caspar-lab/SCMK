"""
run_cardio.py — 纯数值型数据集上的 CKAD 完整消融实验
======================================================

【实验目的】
  在纯数值型数据集（cardio.mat、cardiotocography.mat 等）上评估 CKAD 算法，
  对比四种异常评分器，并扫描隐藏层维度 latent_dim 对结果的影响。

【四种评分器】
  CK-Incon : 跨核不一致性得分——各核嵌入的平均平方距离
  CK-KNN   : 拼接嵌入上的 KNN 平均距离（K 在 [2,60] 遍历取最优）
  OC-SVM   : 线性核单分类 SVM（超参数 nu 在候选集内网格搜索最优 AUC）
  SVDD     : RBF 核单分类 SVM（数学上等价于最小包围超球面 SVDD）

【实验因子】
  latent_dim ∈ {16, 32, 64, 128, 256}  (2^4 ~ 2^8)
  kernel_cfg : 仅使用 Gauss-5-med（数据自适应带宽，消融实验中最稳定的配置）

【为什么只用 Gauss-5-med】
  核消融实验（run_kernel_ablation.py）在多个混合型数据集上对比了 7 种核配置：
    Hetero-5（5种不同核）、Gauss-5-lin（等间隔线性带宽）、
    Gauss-5-log（对数间隔带宽）、Gauss-5-med（中位数自适应带宽）、
    Gauss-2+het、Cauchy-5、Poly-5
  结果表明 Gauss-5-med 在 CK-Incon 指标上整体最稳定，且无需手动调带宽；
  故在主实验中固定使用该配置，避免核选择带来的额外变量。

【Gauss-5-med 带宽计算】
  从正常样本中随机采样 ≤500 个点，计算欧氏距离的中位数 med；
  生成 5 个高斯核，带宽 t = med × r，r ∈ {0.1, 0.5, 1.0, 2.0, 5.0}。
  覆盖从"紧凑"到"宽松"的尺度范围，使模型可以同时捕获局部和全局结构。

【使用方式】
  默认在 cardio.mat 上运行：
    python run_cardio.py
  指定其他数据集：
    python run_cardio.py path/to/dataset.mat

【输出】
  终端：逐维度的四列 AUC 对比表格
  CSV ：../result/{dataset_name}_latent_dim_sweep.csv
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import pairwise_distances
from CKAD import (load_data, train_ckad, get_embeddings,
                  cross_kernel_inconsistency, best_knn_auc,
                  ocsvm_score, svdd_score, roc_auc_score)

# ─── 路径配置 ──────────────────────────────────────────────────────────────────
# 默认数据集路径；也可通过命令行参数传入其他 .mat 文件
_default_path = r'D:\Microsoft\documents\博士课题\异常检测\论文\5\dataset\numerical\cardio.mat'
DATA_PATH  = sys.argv[1] if len(sys.argv) > 1 else _default_path

# 结果目录：脚本所在目录的上一级下的 result 文件夹
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
os.makedirs(RESULT_DIR, exist_ok=True)
_stem      = os.path.splitext(os.path.basename(DATA_PATH))[0]
RESULT_CSV = os.path.join(RESULT_DIR, f'{_stem}_latent_dim_sweep.csv')

# ─── 实验超参数 ────────────────────────────────────────────────────────────────
K_MIN, K_MAX  = 2, 60         # KNN 搜索范围：在此区间内找最优 K（AUC 最高）
LATENT_DIMS   = [16, 32, 64, 128, 256]   # 嵌入维度候选值，对应 2^4 ~ 2^8

# OC-SVM 和 SVDD 的 nu 超参数候选值
# nu 含义：正常训练样本中允许落在决策边界"错误侧"的比例上界
#   nu=0.01 → 极紧边界（几乎不允许误分正常样本），适合异常率极低的数据集
#   nu=0.20 → 较松边界，允许 20% 正常样本作为支持向量，适合噪声较多的数据
# 对每个 nu 分别计算 AUC，取最优值，避免手动调参
NU_CANDIDATES = [0.01, 0.05, 0.1, 0.2]

# 训练配置（所有维度共用，保证公平对比）
TRAIN_CFG = dict(
    epochs     = 100,       # 训练轮数（100 epoch 已足够收敛，更多轮次改善有限）
    batch_size = 512,       # 批大小（需足够大以提供丰富的负对，同时适配内存）
    lr         = 0.01,      # Adam 学习率（较大 lr 配合 batch_size=512 收敛稳定）
    normalize  = True,      # L2 归一化嵌入（统一各核嵌入尺度，改善 OC-SVM 表现）
    seed       = 42,        # 随机种子（保证结果可复现）
)


# ─── 自适应高斯核配置（Gauss-5-med）──────────────────────────────────────────

def gauss_med_kernels(X, ratios=(0.1, 0.5, 1.0, 2.0, 5.0)):
    """
    基于数据自适应带宽生成 5 个高斯核配置。

    带宽估计：
      1. 从 X 中随机采样 min(500, N) 个点（避免大数据集下距离矩阵 OOM）
      2. 计算采样点间欧氏距离矩阵，取中位数 med
      3. 以 med × ratio 为带宽，生成 5 个尺度不同的高斯核

    5 个尺度的作用：
      × 0.1 med → 极窄核，只关注局部最近邻的相似性
      × 0.5 med → 较窄核，捕获局部结构
      × 1.0 med → 标准核（常用"中位数启发式"）
      × 2.0 med → 较宽核，捕获中等尺度结构
      × 5.0 med → 宽核，捕获全局结构

    多尺度核的组合使 CKAD 同时具备局部感知和全局感知能力，
    这正是跨核对比学习（不同尺度核作为不同视图）的优势所在。

    参数：
      X      : (n_normal, D) 正常样本特征矩阵（仅用正常样本估计带宽）
      ratios : 带宽缩放因子序列

    返回：
      kernels : list of (name, 'Gaussian', {'t': bandwidth}) 元组
    """
    rng = np.random.default_rng(0)   # 固定随机种子保证带宽估计可复现
    idx = rng.choice(len(X), min(500, len(X)), replace=False)
    med = np.median(pairwise_distances(X[idx], metric='euclidean'))
    # max(med * r, 1e-3) 防止带宽退化为零（数据过于集中时 med 可能极小）
    return [(f'G-{med*r:.3g}', 'Gaussian', {'t': max(med * r, 1e-3)}) for r in ratios]


# ─── OC-SVM / SVDD 超参数搜索 ─────────────────────────────────────────────────

def best_nu_auc(H_all, H_normal, y, score_fn, nu_list):
    """
    在给定 nu 候选值列表上网格搜索，返回使 ROC-AUC 最高的 (best_nu, best_auc)。

    【为什么需要搜索 nu】
      nu 对 OC-SVM/SVDD 的决策边界松紧程度影响很大，且最优值与数据集
      的异常率和正常样本分布密切相关，难以事先确定。
      通过在有标签的验证集（此处用全量测试集的 AUC 评估）上选择最优 nu，
      确保报告的是每个 nu 下的最佳性能，避免因超参数不当低估算法能力。

      注意：这里用测试集标签选 nu 属于"oracle 评估"，
      实际部署时应在验证集上搜索；此处的目的是评估算法上界。

    参数：
      H_all    : (N, K*d) 全量嵌入（含正常+异常，用于打分）
      H_normal : (n_normal, K*d) 正常嵌入（用于训练 OC-SVM/SVDD）
      y        : (N,) 标签，0=正常，1=异常
      score_fn : 评分函数，接受 (H_all, H_normal, nu=nu) 返回 (N,) 异常得分
                 可传入 ocsvm_score 或 svdd_score
      nu_list  : nu 候选值列表

    返回：
      (best_nu, best_auc) : 最优 nu 值及对应的 ROC-AUC
    """
    best_nu, best_auc = nu_list[0], -1.0
    for nu in nu_list:
        try:
            sc  = score_fn(H_all, H_normal, nu=nu)
            auc = roc_auc_score(y, sc)
            if auc > best_auc:
                best_auc, best_nu = auc, nu
        except Exception:
            # OneClassSVM 在极端 nu 值下可能不收敛，静默跳过
            pass
    return best_nu, best_auc


# ─── 主程序 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'加载: {os.path.basename(DATA_PATH)}')
    X, y, meta = load_data(DATA_PATH)
    N, D, ar  = meta['N'], X.shape[1], meta['anomaly_rate']
    n_normal  = int((y == 0).sum())
    print(f'N={N}  D={D}  异常率={ar*100:.1f}%  正常样本={n_normal}')

    # 用正常样本估计带宽（不让异常样本污染带宽估计）
    kernels = gauss_med_kernels(X[y == 0])
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ── KNN 原始特征基线（只算一次，与 latent_dim 无关）────────────────────────
    # 直接在原始 MinMax 归一化特征上运行 KNN，作为不依赖任何训练的基线。
    # 搜索 K ∈ [2,60]，取 AUC 最高的 K，代表 KNN 的理论最优性能。
    t0 = time.time()
    best_k_knn, auc_knn = best_knn_auc(X, y, K_MIN, K_MAX)
    print(f'KNN 基线: AUC={auc_knn:.4f}  best_k={best_k_knn}  ({time.time()-t0:.1f}s)\n')

    # 表头
    print(f'{"─"*78}')
    print(f'{"dim":>5} {"Incon":>7} {"CK-KNN":>8}{"(k)":>4}  '
          f'{"OC-SVM":>8}{"(nu)":>6}  {"SVDD":>8}{"(nu)":>6}  {"time":>6}')
    print(f'{"─"*78}')

    rows = []
    for latent_dim in LATENT_DIMS:
        # ── 训练 CKAD ────────────────────────────────────────────────────────
        # 每种 latent_dim 独立训练一个新模型（参数量不同，对比才公平）。
        # print_freq=epochs+1 使训练过程静默，避免干扰进度输出。
        t0 = time.time()
        model = train_ckad(
            X, y=y, kernels=kernels, latent_dim=latent_dim,
            device=device, print_freq=TRAIN_CFG['epochs'] + 1,
            **{k: v for k, v in TRAIN_CFG.items()}
        )
        t_train = time.time() - t0

        # ── 提取嵌入 ─────────────────────────────────────────────────────────
        # H_all  : (N, K*d)  — 5 个核的嵌入横向拼接，K=5，d=latent_dim
        # H_per  : list of 5 arrays (N, d) — 各核独立嵌入，用于 CK-Incon
        # H_normal: 正常样本的拼接嵌入，用于训练 OC-SVM/SVDD
        H_all, H_per = get_embeddings(model, X, device=device)
        H_normal = H_all[y == 0]   # 从全量嵌入中取正常样本子集

        # ── 四种异常评分 ──────────────────────────────────────────────────────

        # (a) CK-Incon：跨核不一致性，直接从各核嵌入计算，无需额外训练
        auc_incon = roc_auc_score(y, cross_kernel_inconsistency(H_per))

        # (b) CK-KNN：在拼接嵌入 H_all 上搜索最优 K
        #   注意：H_all 维度 = K * latent_dim，随 latent_dim 变化
        #   维度越高，KNN 受"维度灾难"影响越大，可能导致较大 latent_dim 下 AUC 下降
        bk, auc_cknn = best_knn_auc(H_all, y, K_MIN, K_MAX)

        # (c) OC-SVM（线性核）：在嵌入空间拟合最大间隔超平面
        #   grid-search nu，报告最优 AUC
        best_nu_oc, auc_ocsvm = best_nu_auc(
            H_all, H_normal, y, ocsvm_score, NU_CANDIDATES)

        # (d) SVDD（RBF 核 OC-SVM）：在嵌入空间拟合最小包围超球面
        #   grid-search nu，报告最优 AUC
        best_nu_sv, auc_svdd  = best_nu_auc(
            H_all, H_normal, y, svdd_score,  NU_CANDIDATES)

        # 打印当前维度的结果
        print(f'{latent_dim:>5d} {auc_incon:>7.4f} {auc_cknn:>8.4f}{bk:>4d}  '
              f'{auc_ocsvm:>8.4f}{best_nu_oc:>6.2f}  '
              f'{auc_svdd:>8.4f}{best_nu_sv:>6.2f}  {t_train:>6.1f}s')

        # 记录本维度的完整结果
        rows.append(dict(
            latent_dim    = latent_dim,
            n_train       = n_normal,        # 训练样本数（仅正常样本）
            auc_knn_base  = auc_knn,         # KNN 基线 AUC（与维度无关，每行重复记录）
            best_k_base   = best_k_knn,      # KNN 基线最优 K
            auc_ck_incon  = round(auc_incon, 6),  # CK-Incon AUC
            auc_ck_knn    = round(auc_cknn,  6),  # CK-KNN AUC
            best_k_cknn   = bk,                   # CK-KNN 最优 K
            auc_ocsvm     = round(auc_ocsvm,  6), # OC-SVM AUC（最优 nu 下）
            best_nu_ocsvm = best_nu_oc,           # OC-SVM 最优 nu
            auc_svdd      = round(auc_svdd,   6), # SVDD AUC（最优 nu 下）
            best_nu_svdd  = best_nu_sv,           # SVDD 最优 nu
            train_time_s  = round(t_train, 2),    # 训练耗时（秒）
        ))

    print(f'{"─"*78}')

    # ── 保存 CSV ──────────────────────────────────────────────────────────────
    # 每行对应一个 latent_dim，列包含所有评分器的 AUC 及最优超参数，
    # 便于后续绘图（如 latent_dim vs AUC 折线图）和统计分析
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False)
    print(f'\n结果已保存: {RESULT_CSV}')

    # ── 汇总表：标记每个维度的最优评分器 ───────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'数据集: {_stem}   KNN基线={auc_knn:.4f}')
    print(f'{"dim":>5}  {"Incon":>7}  {"CK-KNN":>7}  {"OC-SVM":>7}  {"SVDD":>7}')
    print(f'{"-"*42}')
    for r in rows:
        # 找出当前维度下四种 CKAD 评分器中的最优 AUC
        best = max(r['auc_ck_incon'], r['auc_ck_knn'],
                   r['auc_ocsvm'], r['auc_svdd'])
        # 最优值前加 * 标记，便于快速定位
        def fmt(v): return f'{"*" if abs(v-best)<1e-6 else " "}{v:.4f}'
        print(f'{r["latent_dim"]:>5d}  {fmt(r["auc_ck_incon"]):>8}  '
              f'{fmt(r["auc_ck_knn"]):>8}  '
              f'{fmt(r["auc_ocsvm"]):>8}  {fmt(r["auc_svdd"]):>8}')
    print(f'{"="*65}')
    print(f'* = 该维度最优评分器')
    print(f'nu候选={NU_CANDIDATES}  K=[{K_MIN},{K_MAX}]  '
          f'epochs={TRAIN_CFG["epochs"]}  batch={TRAIN_CFG["batch_size"]}')
