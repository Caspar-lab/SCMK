"""
CMK 多核异常检测实验
方法对比：KNN-MinMax基线 / 最优单核CMK+KNN / 多核CMK+KNN（5核拼接）
K 在 [K_MIN, K_MAX] 内搜索最优 AUC-ROC
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CMK_AD import run_experiment, ALL_KERNELS

DATA_DIR = r'D:\Microsoft\documents\博士课题\异常检测\论文\5\dataset\mixed'

K_MIN = 2
K_MAX = 60

EXP_CFG = dict(
    latent_dim  = 64,
    epochs      = 100,
    batch_size  = 512,
    lr          = 0.01,
    kernels     = ALL_KERNELS,
    k_min       = K_MIN,
    k_max       = K_MAX,
    normalize   = True,
    seed        = 42,
    verbose     = True,
)

datasets = sorted(f for f in os.listdir(DATA_DIR) if f.endswith('.mat'))

results = []
for fname in datasets:
    path = os.path.join(DATA_DIR, fname)
    print(f'\n{"="*70}')
    print(f'>>> {fname}')
    res = run_experiment(path, **EXP_CFG)
    results.append(res)

# ── 汇总表 ──────────────────────────────────────────────────────────────────
kernels_names = [k[0] for k in ALL_KERNELS]
sep = '=' * 115

print(f'\n\n{sep}')
print(f'AUC-ROC 汇总（K 在 [{K_MIN},{K_MAX}] 内取最优）')
print(f'{"Dataset":<35} {"N":>6} {"Anom%":>6}  '
      f'{"KNN-MM":>7}{"(k)":>4}  '
      f'{"Best-1K":>7}{"(k)":>4}  {"(kernel)":>11}  '
      f'{"MK-CMK":>7}{"(k)":>4}  {"ΔMK-KNN":>8}')
print('-' * 115)

for r in results:
    name = r['dataset'][:34]
    print(f'{name:<35} {r["N"]:>6} {r["anomaly_rate"]*100:>5.1f}%  '
          f'{r["auc_knn"]:>7.4f}({r["best_k_knn"]:>2})  '
          f'{r["auc_best_single"]:>7.4f}({r["best_k_single"]:>2})  '
          f'({r["best_kernel"]:>10})  '
          f'{r["auc_mk"]:>7.4f}({r["best_k_mk"]:>2})  '
          f'{r["auc_mk"]-r["auc_knn"]:>+8.4f}')

auc_knn_avg = sum(r['auc_knn']         for r in results) / len(results)
auc_bs_avg  = sum(r['auc_best_single'] for r in results) / len(results)
auc_mk_avg  = sum(r['auc_mk']          for r in results) / len(results)
print('-' * 115)
print(f'{"Average":<35} {"":>6} {"":>6}  '
      f'{auc_knn_avg:>7.4f}      '
      f'{auc_bs_avg:>7.4f}       {"":>11}  '
      f'{auc_mk_avg:>7.4f}       {auc_mk_avg-auc_knn_avg:>+8.4f}')
print(sep)

# ── 各核单独 AUC 明细 ───────────────────────────────────────────────────────
print(f'\n各核单独 AUC 明细（括号内为最优 K）:')
header = f'{"Dataset":<35}' + ''.join(f'  {k[0]:>14}' for k in ALL_KERNELS) + '  MK-CMK'
print(header)
print('-' * (35 + 16 * len(ALL_KERNELS) + 9))
for r in results:
    row = f'{r["dataset"][:34]:<35}'
    for k in kernels_names:
        auc = r['single_aucs'].get(k, float('nan'))
        bk  = r['single_best_ks'].get(k, 0)
        row += f'  {auc:>7.4f}(k={bk:>2})'
    row += f'  {r["auc_mk"]:>6.4f}'
    print(row)

row_avg = f'{"Average":<35}'
for k in kernels_names:
    avg = sum(r['single_aucs'].get(k, 0) for r in results) / len(results)
    row_avg += f'  {avg:>14.4f}'
row_avg += f'  {auc_mk_avg:>6.4f}'
print('-' * (35 + 16 * len(ALL_KERNELS) + 9))
print(row_avg)

print(f'\nlatent_dim={EXP_CFG["latent_dim"]}  epochs={EXP_CFG["epochs"]}'
      f'  batch={EXP_CFG["batch_size"]}  lr={EXP_CFG["lr"]}'
      f'  K=[{K_MIN},{K_MAX}]')
