"""
temp_sweep_emb.py — per-dataset temperature (tau) sweep with EMBEDDING bandwidths.
================================================================================
The proposed improvement over the original SCMK method combines BOTH loss changes:

  (1) temperature tau        : logits = exp(K_avg / tau)
  (2) embedding bandwidths   : loss Gaussian bandwidths recalibrated each step from
                               the embedding pairwise-distance median (bw_mode='embedding')

For every manuscript dataset EXCEPT pageblocks, on the seed-2 semi-supervised split,
we sweep tau over a grid (training with embedding bandwidths) and report the OPTIMAL
tau per dataset. Each swept model is compared, in AUC, against the ORIGINAL SCMK
method — trained here as the reference cell (raw-feature bandwidths, tau=1, no
temperature), i.e. exactly SCMK's own loss.

Two AUC metrics are recorded (see factorial_legend.md):
  * fused  : max(minmax(emb-lin direction), minmax(mag)) = the ACTUAL SCMK detector.
             This is the HEADLINE comparison ("vs the original SCMK method").
  * emb-MK : Embedding + OC-SVM (pre-computed multi-kernel Gram) — a diagnostic,
             reported alongside for continuity with temp_sweep_all / factorial_all.

Reuses the exact loss/eval code from run_hybrid_score_semi_debug.py.

Run:  conda run -n torch311 python temp_sweep_emb.py
Saves temp_sweep_emb_results.csv (+ prints the aggregate).
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_hybrid_score_semi_debug as dbg   # __main__ guarded -> no side effects

DIM, LAMBDA, SEED = 64, 100.0, 2
BW_MODE = 'embedding'                            # <-- calibrate bandwidth from embedded features
TAUS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]     # temperature grid to scan per dataset
DATASETS = [  # 20 manuscript datasets minus pageblocks
    'vertebral', 'thyroid', 'wbc_malignant_39_variant1', 'glass', 'ecoli',
    'wine', 'cardio', 'cardiotocography_2and3_33_variant1',
    'tic_tac_toe_negative_12_variant1', 'tic_tac_toe_negative_69_variant1', 'wpbc_variant1',
    'ionosphere_b_24_variant1', 'zoo_variant1', 'sick_sick_72_variant1', 'autos_variant1',
    'annealing_variant1', 'lymphography', 'bands_band_6_variant1', 'audiology_variant1',
]
HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), 'result', 'scmk_experiments')
os.makedirs(RESULT_DIR, exist_ok=True)
OUT = os.path.join(RESULT_DIR, 'temp_sweep_emb_results.csv')
SHORT = {'wbc_malignant_39_variant1': 'wbc', 'cardiotocography_2and3_33_variant1': 'cardiotoco',
         'tic_tac_toe_negative_12_variant1': 'tictactoe12', 'tic_tac_toe_negative_69_variant1': 'tictactoe69',
         'wpbc_variant1': 'wpbc', 'ionosphere_b_24_variant1': 'ionosphere', 'zoo_variant1': 'zoo',
         'sick_sick_72_variant1': 'sick', 'autos_variant1': 'autos', 'annealing_variant1': 'annealing',
         'bands_band_6_variant1': 'bands6', 'audiology_variant1': 'audiology'}


def run_dataset(stem, device, cfg):
    X, y, meta = dbg.load_data(dbg.locate(stem))
    yb = (y != 0).astype(int)
    tr, te = dbg.split_indices(yb, SEED)
    y_te = yb[te]
    raw_bw = [k[2]['t'] for k in dbg.gauss_med_kernels(X[tr])]
    raw_rbf, _ = dbg._ocsvm_best(X[te], y_te, X[tr], 'rbf', dbg.NU_LIST)
    rec = dict(dataset=SHORT.get(stem, stem), N=meta['N'], D=X.shape[1], raw_rbf=round(raw_rbf, 4))

    # ORIGINAL SCMK reference: raw-feature bandwidths, tau=1, no temperature.
    m0, _ = dbg.train_variant(X[tr], dbg.RATIOS, raw_bw, DIM, device, LAMBDA, 1.0, 'raw', cfg, False)
    r0 = dbg.eval_all(m0, X, tr, te, y_te, device)
    rec['scmk_fused'] = round(r0['fused'], 4)
    rec['scmk_embmk'] = round(r0['emb_mk'], 4)

    # temperature sweep, EMBEDDING bandwidths.
    for tau in TAUS:
        model, _ = dbg.train_variant(X[tr], dbg.RATIOS, raw_bw, DIM, device, LAMBDA, tau, BW_MODE, cfg, False)
        r = dbg.eval_all(model, X, tr, te, y_te, device)
        rec[f'fused_t{tau}'] = round(r['fused'], 4)
        rec[f'embmk_t{tau}'] = round(r['emb_mk'], 4)
    return rec


def _best(rec, prefix):
    """Return (best_tau, best_auc) over the tau grid for metric `prefix`."""
    vals = [rec[f'{prefix}_t{t}'] for t in TAUS]
    bi = int(np.argmax(vals))
    return TAUS[bi], vals[bi]


def aggregate(df):
    banner = '=' * 96
    print('\n' + banner)
    print(f'TEMPERATURE SWEEP w/ EMBEDDING bandwidth  (dim={DIM}, lam={LAMBDA}, seed={SEED}, '
          f'{len(df)} datasets, pageblocks excluded)')
    print('  HEADLINE metric = fused (the actual SCMK detector); emb-MK shown alongside.')
    print(banner)

    # ---- per-dataset table on the HEADLINE metric (fused) ----
    hdr = (f'\n  {"dataset":<12}{"raw":>7}{"SCMK":>8}' +
           ''.join(f'{("t"+str(t)):>7}' for t in TAUS) + f'{"best_t":>8}{"best":>7}{"gain":>7}')
    print(hdr)
    bt_f, best_f, orig_f = [], [], []
    for _, r in df.iterrows():
        bt, bv = _best(r, 'fused')
        bt_f.append(bt); best_f.append(bv); orig_f.append(r['scmk_fused'])
        row = (f'  {r["dataset"]:<12}{r["raw_rbf"]:>7.3f}{r["scmk_fused"]:>8.3f}' +
               ''.join((f'{r[f"fused_t"+str(t)]:>7.3f}' if t != bt else f'{("*%.3f"%r[f"fused_t"+str(t)]):>7}')
                       for t in TAUS))
        print(row + f'{bt:>8}{bv:>7.3f}{bv-r["scmk_fused"]:>+7.3f}')

    bt_f, best_f, orig_f = np.array(bt_f), np.array(best_f), np.array(orig_f)
    print('\n  per-dataset BEST tau distribution (by fused):')
    for t in TAUS:
        n = int((bt_f == t).sum())
        if n:
            print(f'    tau={t:<5} best for {n:2d} dataset(s)' + ('   <- tau=1 (no temperature)' if t == 1.0 else ''))

    # ---- summary on BOTH metrics ----
    def summarize(prefix, orig_col, label):
        best_pd = np.array([_best(r, prefix)[1] for _, r in df.iterrows()])
        orig = df[orig_col].values
        glob = {t: df[f'{prefix}_t{t}'].mean() for t in TAUS}
        gbt = max(glob, key=glob.get)
        print(f'\n  [{label}]  (mean AUC over {len(df)} datasets)')
        print(f'    original SCMK (raw bw, tau=1)   : {orig.mean():.4f}')
        print(f'    emb-bw, best single global tau={gbt:<4}: {glob[gbt]:.4f}   ({glob[gbt]-orig.mean():+.4f} vs SCMK)')
        print(f'    emb-bw, per-dataset ORACLE tau  : {best_pd.mean():.4f}   ({best_pd.mean()-orig.mean():+.4f} vs SCMK)  [tuning upper bound]')
        win = int((best_pd > orig + 0.005).sum()); lose = int((best_pd < orig - 0.005).sum())
        print(f'    oracle-tau vs SCMK: win {win:2d} / tie {len(df)-win-lose:2d} / lose {lose:2d}')
        return gbt

    print('\n' + '-' * 96)
    summarize('fused', 'scmk_fused', 'FUSED = SCMK detector (HEADLINE)')
    summarize('embmk', 'scmk_embmk', 'emb-MK diagnostic')
    print(f'\n  raw-RBF baseline (loss-independent) mean = {df["raw_rbf"].mean():.4f}')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = {**dbg._BASE_CFG, 'lambda_scatter': LAMBDA}
    print(f'device={device}  datasets={len(DATASETS)} (pageblocks excluded)  '
          f'bw={BW_MODE}  taus={TAUS}', flush=True)
    rows, t0 = [], time.time()
    for i, stem in enumerate(DATASETS, 1):
        try:
            rec = run_dataset(stem, device, cfg)
            rows.append(rec)
            bt, bv = _best(rec, 'fused')
            print(f'[{i:2d}/{len(DATASETS)}] {rec["dataset"]:<12} raw={rec["raw_rbf"]:.3f}  '
                  f'SCMK={rec["scmk_fused"]:.3f}  best(t={bt})={bv:.3f}  ({bv-rec["scmk_fused"]:+.3f})', flush=True)
        except Exception as e:
            print(f'[{i:2d}/{len(DATASETS)}] {stem} ERROR {e}', flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)   # incremental save
    aggregate(pd.DataFrame(rows))
    print(f'\nsaved: {OUT}   (elapsed {(time.time()-t0)/60:.1f} min)')
