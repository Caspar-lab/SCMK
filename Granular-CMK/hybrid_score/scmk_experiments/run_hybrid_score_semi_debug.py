"""
run_hybrid_score_semi_debug.py — DEBUG / TRACE + A/B copy of run_hybrid_score_semi.py
=====================================================================================
Two additions over the real pipeline, to test a hypothesis about the cross-kernel
contrastive loss (CMK_OCSVM.cross_kernel_loss):

  (1) TEMPERATURE tau:  logits = exp(K_avg / tau)   (original == tau=1, no temperature)
  (2) EMBEDDING bandwidths: recalibrate the Gaussian bandwidths from the *embedding*
      pairwise-distance median each step, instead of the raw-feature median that
      gauss_med_kernels() produces (the original applies raw-space bandwidths to
      L2-normalised embeddings on the unit sphere -> extreme kernels dead/saturated).

It then runs an A/B: train with the ORIGINAL loss vs the FIXED loss and compare, under
the identical semi-supervised seed split,

  * standard OCSVM        : RBF OC-SVM on the RAW features            (loss-independent)
  * embedding + OCSVM      : (a) linear OC-SVM on the concat embeddings (= direction signal)
    (pre-computed)           (b) pre-computed multi-kernel Gaussian OC-SVM on the embeddings
  * SCMK dual-signal       : direction / magnitude / max-fused

Question answered: do the loss issues cause "embedding + OCSVM" to fail to beat the
plain raw OCSVM, and does fixing the loss close/flip the gap?

**Nothing is written to disk.**  Run:
  conda run -n torch311 python run_hybrid_score_semi_debug.py [dataset_stem_or_.mat]
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch                                  # before sklearn (Windows OpenMP order)
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, pairwise_distances

_HS = os.path.dirname(os.path.abspath(__file__))
# this module now lives in Granular-CMK/hybrid_score/scmk_experiments/, so CMK_OCSVM
# (in Granular-CMK/) is two levels up; add both parent and grandparent for robustness.
sys.path.insert(0, os.path.dirname(_HS))            # hybrid_score/
sys.path.insert(0, os.path.dirname(os.path.dirname(_HS)))   # Granular-CMK/ on path
from CMK_OCSVM import (load_data, gauss_med_kernels, CMKNet,
                       _kernel_mat, NU_CANDIDATES, TRAIN_CFG as _BASE_CFG)
from CMK_OCSVM_scatter import scatter_loss

# ── CONFIG (edit freely) ──────────────────────────────────────────────────────
DATASET   = sys.argv[1] if len(sys.argv) > 1 else 'cardio'
DIM       = 64
LAMBDA    = 100.0        # scatter weight
SEED      = 2
TEMPERATURE = 0.2        # tau for the FIXED variant (original uses tau=1)
RATIOS    = (0.1, 0.5, 1.0, 2.0, 5.0)   # same multi-scale ratios as gauss_med_kernels
TRAIN_FRAC = 0.5
NU_LIST   = NU_CANDIDATES
VERBOSE_TRAIN = False    # per-epoch loss printing
SEARCH_DIRS = [r'C:/OD/Shihao/datasets', r'C:/OD/Shihao/5/dataset/numerical',
               r'C:/OD/Shihao/5/dataset/nominal', r'C:/OD/Shihao/5/dataset/mixed']


def banner(m): print(f'\n{"="*76}\n{m}\n{"="*76}', flush=True)


def locate(name):
    if name.lower().endswith('.mat') and os.path.exists(name):
        return name
    for d in SEARCH_DIRS:
        p = os.path.join(d, name + '.mat')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'dataset not found: {name}')


def split_indices(y, seed, frac=TRAIN_FRAC):
    rng = np.random.default_rng(seed)
    normal = np.where(y == 0)[0]; anom = np.where(y != 0)[0]
    perm = rng.permutation(normal); k = int(round(len(perm) * frac))
    return perm[:k], np.concatenate([perm[k:], anom])


@torch.no_grad()
def extract_components(model, X, device, batch_size=2048):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    per = [[] for _ in model.projectors]
    for i in range(0, len(X), batch_size):
        xb = X_t[i:i + batch_size].to(device)
        for k, h in enumerate([p(xb) for p in model.projectors]):
            per[k].append(h.cpu().numpy())
    H_raw_per = [np.concatenate(p, axis=0) for p in per]
    H_norm_per = [h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8) for h in H_raw_per]
    H_norms = np.concatenate([np.linalg.norm(h, axis=1, keepdims=True) for h in H_raw_per], axis=1)
    return H_norm_per, H_norms


def _minmax(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-8) if hi > lo else np.zeros_like(s)


def _ocsvm_best(H_te, y_te, H_tr, kernel, nu_list):
    """Semi-supervised OC-SVM: fit on train-normal, score test; best AUC over nu."""
    best, best_s = -1.0, None
    for nu in nu_list:
        try:
            clf = OneClassSVM(kernel=kernel, nu=nu).fit(H_tr)
            s = -clf.decision_function(H_te)
            a = roc_auc_score(y_te, s)
            if a > best:
                best, best_s = a, s
        except Exception:
            pass
    return best, best_s


# ── modified cross-kernel loss: temperature + bandwidth mode ──────────────────
def _median_dist(F, max_n=256):
    n = F.shape[0]
    if n > max_n:
        F = F[torch.randperm(n, device=F.device)[:max_n]]
    # bandwidth is a per-step calibration constant -> detach (no grad through it)
    return torch.pdist(F.detach()).median().clamp(min=1e-3)


def cross_kernel_loss_cfg(hs, ratios, raw_bw, tau=1.0, bw_mode='raw'):
    """Cross-kernel InfoNCE with temperature `tau` and bandwidth mode:
       bw_mode='raw'       -> raw-feature bandwidths (original behaviour)
       bw_mode='embedding' -> bandwidths = median(embedding dist) * ratio, per step."""
    K, B = len(hs), hs[0].shape[0]
    dev = hs[0].device
    mask = torch.eye(B, device=dev).repeat(2, 2) * (1 - torch.eye(2 * B, device=dev))
    logits_mask = 1 - torch.eye(2 * B, device=dev)
    total, n = 0.0, 0
    for k in range(K):
        for l in range(k + 1, K):
            F = torch.cat([hs[k], hs[l]], dim=0)
            if bw_mode == 'embedding':
                med = _median_dist(F)
                tk, tl = med * ratios[k], med * ratios[l]
            else:
                tk, tl = raw_bw[k], raw_bw[l]
            K_avg = (_kernel_mat(F, 'Gaussian', {'t': float(tk)}) +
                     _kernel_mat(F, 'Gaussian', {'t': float(tl)})) / 2
            logits = torch.exp(K_avg / tau)
            log_prob = torch.log(logits) - torch.log((logits * logits_mask).sum(1, keepdim=True))
            loss = -(mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
            total += loss.mean(); n += 1
    return total / max(n, 1)


def train_variant(X_tr, ratios, raw_bw, dim, device, lam, tau, bw_mode, cfg, verbose=False):
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed'])
    N, D = X_tr.shape
    model = CMKNet(D, dim, len(ratios), cfg['normalize']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    X_t = torch.tensor(X_tr, dtype=torch.float32)
    last = 0.0
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        perm = torch.randperm(N)
        ec = es = et = 0.0; nb = 0
        for i in range(0, N, cfg['batch_size']):
            idx = perm[i:i + cfg['batch_size']]
            if len(idx) < 4:
                continue
            hs = model(X_t[idx].to(device))
            Lc = cross_kernel_loss_cfg(hs, ratios, raw_bw, tau=tau, bw_mode=bw_mode)
            Ls = scatter_loss(hs) if lam > 0 else torch.zeros((), device=device)
            loss = Lc + lam * Ls
            opt.zero_grad(); loss.backward(); opt.step()
            ec += Lc.item(); es += float(Ls.item()); et += loss.item(); nb += 1
        last = et / max(nb, 1)
        if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == cfg['epochs']):
            print(f'      epoch {epoch:>3d}  L_cross={ec/nb:.4f}  L_scatter={es/nb:+.4f}  total={et/nb:.4f}')
    return model, last


def emb_mk_precomp_auc(H_norm_per, ratios, tr, te, y_te, nu_list):
    """Embedding + OC-SVM (pre-computed): averaged multi-kernel Gaussian Gram on the
       LEARNED per-head embeddings, bandwidth = median(embedding dist) * ratio."""
    K, N = len(H_norm_per), H_norm_per[0].shape[0]
    Gram = np.zeros((N, N))
    for k in range(K):
        d2 = pairwise_distances(H_norm_per[k], metric='sqeuclidean')
        med = np.sqrt(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
        t = med * ratios[k]
        Gram += np.exp(-d2 / (2.0 * t * t))
    Gram /= K
    Kfit, Kall = Gram[np.ix_(tr, tr)], Gram[np.ix_(te, tr)]
    best = -1.0
    for nu in nu_list:
        try:
            clf = OneClassSVM(kernel='precomputed', nu=nu).fit(Kfit)
            best = max(best, roc_auc_score(y_te, -clf.decision_function(Kall)))
        except Exception:
            pass
    return best


def eval_all(model, X, tr, te, y_te, device):
    H_norm_per, H_norms = extract_components(model, X, device)
    H_dir = np.concatenate(H_norm_per, axis=1)
    a_dir, s_dir = _ocsvm_best(H_dir[te], y_te, H_dir[tr], 'linear', NU_LIST)   # emb-linear (direction)
    a_mag, s_mag = _ocsvm_best(H_norms[te], y_te, H_norms[tr], 'rbf', NU_LIST)  # magnitude
    if s_dir is None or s_mag is None:
        fused = np.nan
    else:
        fused = roc_auc_score(y_te, np.maximum(_minmax(s_dir), _minmax(s_mag)))
    a_embmk = emb_mk_precomp_auc(H_norm_per, RATIOS, tr, te, y_te, NU_LIST)     # emb multi-kernel precomp
    return dict(emb_lin=a_dir, emb_mk=a_embmk, mag=a_mag, fused=fused)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = {**_BASE_CFG, 'lambda_scatter': LAMBDA}
    banner(f'A/B DEBUG  dataset={DATASET}  dim={DIM}  lambda={LAMBDA}  seed={SEED}  '
           f'tau_fixed={TEMPERATURE}  device={device}')
    print('trace-only run — no files written.')

    X, y, meta = load_data(locate(DATASET))
    yb = (y != 0).astype(int)
    tr, te = split_indices(yb, SEED)
    y_te = yb[te]
    print(f'N={meta["N"]} D={X.shape[1]} anomaly={meta["anomaly_rate"]*100:.1f}%  '
          f'train(normal)={len(tr)} test={len(te)} ({int((y_te==0).sum())} normal + {int(y_te.sum())} anom)')

    kernels = gauss_med_kernels(X[tr])
    raw_bw = [k[2]['t'] for k in kernels]

    # concretely show the raw-vs-embedding bandwidth mismatch
    banner('bandwidth mismatch check (raw-feature bandwidths vs embedding geometry)')
    with torch.no_grad():
        m0 = CMKNet(X.shape[1], DIM, len(RATIOS), cfg['normalize']).to(device)
        h0 = m0(torch.tensor(X[tr], dtype=torch.float32).to(device))[0]
        emb_med = float(torch.pdist(h0[:min(256, len(h0))]).median())
    print(f'  raw-feature bandwidths t_k        : {[round(b,3) for b in raw_bw]}')
    print(f'  median EMBEDDING pairwise distance : {emb_med:.3f}  (unit sphere, dist in [0,2])')
    print(f'  -> emb-calibrated bandwidths would : {[round(emb_med*r,3) for r in RATIOS]}')
    for b, r in zip(raw_bw, RATIOS):
        k_at_med = np.exp(-(emb_med**2) / (2 * b * b))
        flag = 'DEAD (~0)' if k_at_med < 0.02 else ('SATURATED (~1)' if k_at_med > 0.9 else 'ok')
        print(f'     ratio {r:<4}: raw t={b:.3f} -> Gaussian@median={k_at_med:.3f}  [{flag}]')

    # standard OCSVM on raw features (loss-independent baseline)
    raw_rbf, _ = _ocsvm_best(X[te], y_te, X[tr], 'rbf', NU_LIST)

    # 2x2 factorial: {raw bw, embedding bw} x {tau=1, tau=TEMPERATURE}
    #   -> isolates the effect of EACH change (and their interaction).
    T = TEMPERATURE
    CELLS = [('raw', 1.0), ('raw', T), ('embedding', 1.0), ('embedding', T)]
    LABEL = {('raw', 1.0): 'raw bw, tau=1     (original)',
             ('raw', T):   f'raw bw, tau={T}   (TEMP only)',
             ('embedding', 1.0): 'emb bw, tau=1     (BW only)',
             ('embedding', T):   f'emb bw, tau={T}   (BOTH)'}
    res = {}
    for bw, tau in CELLS:
        banner(f'train + eval  —  {LABEL[(bw, tau)]}')
        t0 = time.time()
        model, last = train_variant(X[tr], RATIOS, raw_bw, DIM, device, LAMBDA, tau, bw, cfg, VERBOSE_TRAIN)
        r = eval_all(model, X, tr, te, y_te, device)
        res[(bw, tau)] = r
        print(f'  final total loss={last:.4f}  ({time.time()-t0:.1f}s)   '
              f'emb-lin={r["emb_lin"]:.4f}  emb-MK={r["emb_mk"]:.4f}  mag={r["mag"]:.4f}  fused={r["fused"]:.4f}')

    # summary: factorial table + marginal (main) effects on emb-MK
    banner('2x2 FACTORIAL — emb-MK precomp AUC  (isolating each change)')
    print(f'  standard OCSVM (raw RBF, loss-independent): {raw_rbf:.4f}\n')
    print(f'  {"":14}{"tau=1":>12}{"tau="+str(T):>12}')
    for bw in ('raw', 'embedding'):
        print(f'  {bw+" bw":<14}{res[(bw,1.0)]["emb_mk"]:>12.4f}{res[(bw,T)]["emb_mk"]:>12.4f}')
    e = lambda bw, tau: res[(bw, tau)]['emb_mk']
    dT_raw = e('raw', T) - e('raw', 1.0)
    dT_emb = e('embedding', T) - e('embedding', 1.0)
    dB_t1 = e('embedding', 1.0) - e('raw', 1.0)
    dB_tT = e('embedding', T) - e('raw', T)
    inter = dT_emb - dT_raw
    print('\n  main effects on emb-MK AUC:')
    print(f'    temperature  (holding bw): raw bw {dT_raw:+.4f}   |   emb bw {dT_emb:+.4f}')
    print(f'    bandwidth    (holding tau): tau=1 {dB_t1:+.4f}   |   tau={T} {dB_tT:+.4f}')
    print(f'    interaction (temp x bw)  : {inter:+.4f}')
    print(f'    total (both vs original) : {e("embedding",T)-e("raw",1.0):+.4f}')
    banner('DONE (no files written)')


if __name__ == '__main__':
    main()
