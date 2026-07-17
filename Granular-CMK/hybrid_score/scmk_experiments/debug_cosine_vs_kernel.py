"""
debug_cosine_vs_kernel.py — cosine NT-Xent loss  vs  SCMK's kernel-logit loss.
================================================================================
Replaces the cross-kernel contrastive logits with the standard NT-Xent form
    logits = (h_k . h_l) / tau          (cosine similarity of L2-normalised
                                         embeddings, fixed temperature tau ~ 0.1)
and compares it against the current SCMK loss (Gaussian-kernel value as the logit,
raw-feature bandwidths, no temperature). Everything else is identical: K=5 heads,
lambda*scatter, Adam, 100 epochs, seed-2 split, dim=64.

Both losses are scored with the same detectors (via run_hybrid_score_semi_debug):
  emb-lin (= SCMK direction) | emb-MK precomp | fused (= SCMK detector) | vs raw-RBF.

Usage:
  conda run -n torch311 python debug_cosine_vs_kernel.py [dataset]   # single-dataset smoke test
  conda run -n torch311 python debug_cosine_vs_kernel.py             # full 19-dataset aggregate
Saves cosine_vs_kernel_results.csv on the full run.
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_hybrid_score_semi_debug as dbg
from CMK_OCSVM import cross_kernel_loss   # the ORIGINAL SCMK kernel-logit loss

DIM, LAMBDA, SEED = 64, 100.0, 2
COS_TAUS = [0.07, 0.1, 0.2]
DATASETS = [
    'vertebral', 'thyroid', 'wbc_malignant_39_variant1', 'glass', 'ecoli',
    'wine', 'cardio', 'cardiotocography_2and3_33_variant1',
    'tic_tac_toe_negative_12_variant1', 'tic_tac_toe_negative_69_variant1', 'wpbc_variant1',
    'ionosphere_b_24_variant1', 'zoo_variant1', 'sick_sick_72_variant1', 'autos_variant1',
    'annealing_variant1', 'lymphography', 'bands_band_6_variant1', 'audiology_variant1',
]
SHORT = {'wbc_malignant_39_variant1': 'wbc', 'cardiotocography_2and3_33_variant1': 'cardiotoco',
         'tic_tac_toe_negative_12_variant1': 'tictactoe12', 'tic_tac_toe_negative_69_variant1': 'tictactoe69',
         'wpbc_variant1': 'wpbc', 'ionosphere_b_24_variant1': 'ionosphere', 'zoo_variant1': 'zoo',
         'sick_sick_72_variant1': 'sick', 'autos_variant1': 'autos', 'annealing_variant1': 'annealing',
         'bands_band_6_variant1': 'bands6', 'audiology_variant1': 'audiology'}
HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), 'result', 'scmk_experiments')
os.makedirs(RESULT_DIR, exist_ok=True)
OUT = os.path.join(RESULT_DIR, 'cosine_vs_kernel_results.csv')


def cross_kernel_loss_cosine(hs, tau):
    """Standard NT-Xent over the K heads: logits = cosine / tau, same positive
       (same sample across two heads), logsumexp-stable softmax."""
    K, B = len(hs), hs[0].shape[0]
    dev = hs[0].device
    eye2 = torch.eye(2 * B, device=dev)
    mask = torch.eye(B, device=dev).repeat(2, 2) * (1 - eye2)   # positive = (i, i+B)
    total, n = 0.0, 0
    for k in range(K):
        for l in range(k + 1, K):
            F = torch.cat([hs[k], hs[l]], dim=0)                # (2B, d), already L2-normalised
            logits = (F @ F.T) / tau
            logits = logits.masked_fill(eye2.bool(), -1e9)      # exclude self from the softmax
            log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            loss = -(mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
            total += loss.mean(); n += 1
    return total / max(n, 1)


def train_with_loss(X_tr, dim, K, device, lam, cfg, loss_fn):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    N, D = X_tr.shape
    model = dbg.CMKNet(D, dim, K, cfg['normalize']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    X_t = torch.tensor(X_tr, dtype=torch.float32)
    for _ in range(cfg['epochs']):
        model.train()
        perm = torch.randperm(N)
        for i in range(0, N, cfg['batch_size']):
            idx = perm[i:i + cfg['batch_size']]
            if len(idx) < 4:
                continue
            hs = model(X_t[idx].to(device))
            Lc = loss_fn(hs)
            Ls = dbg.scatter_loss(hs) if lam > 0 else torch.zeros((), device=device)
            loss = Lc + lam * Ls
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def run_dataset(stem, device, cfg, verbose=False):
    X, y, meta = dbg.load_data(dbg.locate(stem))
    yb = (y != 0).astype(int)
    tr, te = dbg.split_indices(yb, SEED)
    y_te = yb[te]
    kernels = dbg.gauss_med_kernels(X[tr])
    K = len(kernels)
    raw_rbf, _ = dbg._ocsvm_best(X[te], y_te, X[tr], 'rbf', dbg.NU_LIST)
    rec = dict(dataset=SHORT.get(stem, stem), raw_rbf=round(raw_rbf, 4))

    variants = [('kernel', lambda hs: cross_kernel_loss(hs, kernels))]
    for t in COS_TAUS:
        variants.append((f'cos{t}', (lambda tt: (lambda hs: cross_kernel_loss_cosine(hs, tt)))(t)))
    for name, fn in variants:
        model = train_with_loss(X[tr], DIM, K, device, LAMBDA, cfg, fn)
        r = dbg.eval_all(model, X, tr, te, y_te, device)
        rec[f'{name}_emblin'] = round(r['emb_lin'], 4)
        rec[f'{name}_embmk'] = round(r['emb_mk'], 4)
        rec[f'{name}_fused'] = round(r['fused'], 4)
        if verbose:
            print(f'  {name:<8} emb-lin={r["emb_lin"]:.4f}  emb-MK={r["emb_mk"]:.4f}  '
                  f'mag={r["mag"]:.4f}  fused={r["fused"]:.4f}')
    return rec


def aggregate(df):
    variants = ['kernel'] + [f'cos{t}' for t in COS_TAUS]
    print('\n' + '=' * 84)
    print(f'COSINE NT-Xent vs KERNEL-LOGIT loss  (dim={DIM}, lam={LAMBDA}, seed={SEED}, {len(df)} datasets)')
    print('=' * 84)
    print(f'\n  raw-RBF baseline (mean): {df["raw_rbf"].mean():.4f}\n')
    print(f'  {"loss":<10}{"emb-lin":>9}{"emb-MK":>9}{"fused":>9}   (mean AUC over datasets)')
    for v in variants:
        print(f'  {v:<10}{df[f"{v}_emblin"].mean():>9.4f}{df[f"{v}_embmk"].mean():>9.4f}{df[f"{v}_fused"].mean():>9.4f}')

    print('\n  cosine vs kernel-logit  (per-dataset, win >+0.005 / tie / lose):')
    for metric in ['emblin', 'embmk', 'fused']:
        print(f'    [{metric}]')
        for t in COS_TAUS:
            d = df[f'cos{t}_{metric}'] - df[f'kernel_{metric}']
            w = int((d > 0.005).sum()); l = int((d < -0.005).sum()); ti = len(df) - w - l
            print(f'      cos{t} vs kernel : mean {d.mean():+.4f}   win {w:2d} / tie {ti:2d} / lose {l:2d}')

    # vs raw-RBF on the SCMK detector (fused) and on emb-MK
    print('\n  beats raw-RBF (win >+0.005 / tie / lose):')
    for metric in ['fused', 'embmk']:
        print(f'    [{metric}]')
        for v in variants:
            d = df[f'{v}_{metric}'] - df['raw_rbf']
            w = int((d > 0.005).sum()); l = int((d < -0.005).sum()); ti = len(df) - w - l
            print(f'      {v:<8} mean gap {d.mean():+.4f}   win {w:2d} / tie {ti:2d} / lose {l:2d}')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = {**dbg._BASE_CFG, 'lambda_scatter': LAMBDA}
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only:
        print(f'device={device}  SMOKE TEST on {only}  (cos taus={COS_TAUS})')
        rec = run_dataset(only, device, cfg, verbose=True)
        print(f'  raw-RBF={rec["raw_rbf"]:.4f}')
    else:
        print(f'device={device}  datasets={len(DATASETS)}  variants=kernel+{["cos"+str(t) for t in COS_TAUS]}', flush=True)
        rows, t0 = [], time.time()
        for i, stem in enumerate(DATASETS, 1):
            try:
                rec = run_dataset(stem, device, cfg)
                rows.append(rec)
                print(f'[{i:2d}/{len(DATASETS)}] {rec["dataset"]:<12} raw={rec["raw_rbf"]:.3f}  ' +
                      f'kernel(fused)={rec["kernel_fused"]:.3f}  ' +
                      '  '.join(f'cos{t}(fused)={rec[f"cos{t}_fused"]:.3f}' for t in COS_TAUS), flush=True)
            except Exception as e:
                print(f'[{i:2d}/{len(DATASETS)}] {stem} ERROR {e}', flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
        aggregate(pd.DataFrame(rows))
        print(f'\nsaved: {OUT}  (elapsed {(time.time()-t0)/60:.1f} min)')
