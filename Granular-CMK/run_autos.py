import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CMK_AD import run_experiment, ALL_KERNELS

path = r'D:\Microsoft\documents\博士课题\异常检测\论文\5\dataset\mixed\autos_variant1.mat'

res = run_experiment(
    path,
    latent_dim  = 64,
    epochs      = 100,
    batch_size  = 512,
    lr          = 0.01,
    kernels     = ALL_KERNELS,
    k_min       = 2,
    k_max       = 60,
    normalize   = True,
    seed        = 42,
    verbose     = True,
)

kernels_names = [k[0] for k in ALL_KERNELS]
print()
print(f'{"="*60}')
print(f'Dataset : {res["dataset"]}')
print(f'N={res["N"]}  anomaly_rate={res["anomaly_rate"]*100:.1f}%')
print(f'KNN-MinMax  : AUC={res["auc_knn"]:.4f}  best_k={res["best_k_knn"]}')
for k in kernels_names:
    print(f'  [{k:<11}]: AUC={res["single_aucs"][k]:.4f}  best_k={res["single_best_ks"][k]}')
print(f'Best Single : AUC={res["auc_best_single"]:.4f}  kernel={res["best_kernel"]}  k={res["best_k_single"]}')
print(f'Multi-Kernel: AUC={res["auc_mk"]:.4f}  best_k={res["best_k_mk"]}')
print(f'Delta(MK-KNN): {res["auc_mk"]-res["auc_knn"]:+.4f}')
print(f'{"="*60}')
