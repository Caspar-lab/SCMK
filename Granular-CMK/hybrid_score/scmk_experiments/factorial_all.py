"""
factorial_all.py — 2x2 factorial (bandwidth x temperature) across all datasets.
================================================================================
For every manuscript dataset EXCEPT pageblocks (the largest), on the seed-2 split,
train the SCMK encoder under the 4 loss settings

    {raw bw, embedding bw} x {tau=1 (original), tau=0.2}

and evaluate "embedding + OC-SVM (pre-computed)" (emb-MK precomp) against the
loss-independent standard raw-RBF OC-SVM. Reuses the exact loss/eval code from
run_hybrid_score_semi_debug.py. Aggregates main effects (temperature vs bandwidth)
over datasets.

Run:  conda run -n torch311 python factorial_all.py
Saves per-dataset results to factorial_all_results.csv (+ prints the aggregate).
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_hybrid_score_semi_debug as dbg   # __main__ guarded -> no side effects

DIM, LAMBDA, SEED, T = 64, 100.0, 2, 0.2
CELLS = [('raw', 1.0), ('raw', T), ('embedding', 1.0), ('embedding', T)]
NAME = {('raw', 1.0): 'orig(raw,t1)', ('raw', T): 'temp(raw,t.2)',
        ('embedding', 1.0): 'bw(emb,t1)', ('embedding', T): 'both(emb,t.2)'}

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
OUT = os.path.join(RESULT_DIR, 'factorial_all_results.csv')


def run_dataset(stem, device, cfg):
    X, y, meta = dbg.load_data(dbg.locate(stem))
    yb = (y != 0).astype(int)
    tr, te = dbg.split_indices(yb, SEED)
    y_te = yb[te]
    kernels = dbg.gauss_med_kernels(X[tr])
    raw_bw = [k[2]['t'] for k in kernels]
    raw_rbf, _ = dbg._ocsvm_best(X[te], y_te, X[tr], 'rbf', dbg.NU_LIST)
    rec = dict(dataset=stem, N=meta['N'], D=X.shape[1], raw_rbf=round(raw_rbf, 4))
    for bw, tau in CELLS:
        model, _ = dbg.train_variant(X[tr], dbg.RATIOS, raw_bw, DIM, device, LAMBDA, tau, bw, cfg, False)
        r = dbg.eval_all(model, X, tr, te, y_te, device)
        tag = NAME[(bw, tau)]
        rec[f'{tag}_embmk'] = round(r['emb_mk'], 4)
        rec[f'{tag}_emblin'] = round(r['emb_lin'], 4)
        rec[f'{tag}_fused'] = round(r['fused'], 4)
    return rec


def aggregate(df):
    tags = [NAME[c] for c in CELLS]
    print('\n' + '=' * 78)
    print(f'AGGREGATE over {len(df)} datasets  (metric: emb-MK precomp AUC; seed=2, dim={DIM}, lam={LAMBDA}, tau={T})')
    print('=' * 78)
    m = {c: df[f'{NAME[c]}_embmk'].mean() for c in CELLS}
    print(f'\n  standard raw-RBF OCSVM (baseline) mean = {df["raw_rbf"].mean():.4f}\n')
    print(f'  mean emb-MK AUC        {"tau=1":>10}{"tau="+str(T):>10}')
    print(f'    raw bw            {m[("raw",1.0)]:>10.4f}{m[("raw",T)]:>10.4f}')
    print(f'    embedding bw      {m[("embedding",1.0)]:>10.4f}{m[("embedding",T)]:>10.4f}')

    # per-dataset main effects -> mean +/- std
    dT_raw = df[f'{NAME[("raw",T)]}_embmk'] - df[f'{NAME[("raw",1.0)]}_embmk']
    dT_emb = df[f'{NAME[("embedding",T)]}_embmk'] - df[f'{NAME[("embedding",1.0)]}_embmk']
    dB_t1 = df[f'{NAME[("embedding",1.0)]}_embmk'] - df[f'{NAME[("raw",1.0)]}_embmk']
    dB_tT = df[f'{NAME[("embedding",T)]}_embmk'] - df[f'{NAME[("raw",T)]}_embmk']
    inter = dT_emb - dT_raw
    tot = df[f'{NAME[("embedding",T)]}_embmk'] - df[f'{NAME[("raw",1.0)]}_embmk']
    print('\n  main effects on emb-MK AUC (mean +/- std over datasets):')
    print(f'    TEMPERATURE | raw bw : {dT_raw.mean():+.4f} +/- {dT_raw.std():.4f}')
    print(f'    TEMPERATURE | emb bw : {dT_emb.mean():+.4f} +/- {dT_emb.std():.4f}')
    print(f'    BANDWIDTH   | tau=1  : {dB_t1.mean():+.4f} +/- {dB_t1.std():.4f}')
    print(f'    BANDWIDTH   | tau={T} : {dB_tT.mean():+.4f} +/- {dB_tT.std():.4f}')
    print(f'    interaction (TxB)    : {inter.mean():+.4f}')
    print(f'    TOTAL (both vs orig) : {tot.mean():+.4f} +/- {tot.std():.4f}')

    # vs raw baseline: win / tie / lose counts per cell
    print('\n  emb-MK vs standard raw-RBF OCSVM  (win >+0.005 / tie / lose <-0.005):')
    for c in CELLS:
        gap = df[f'{NAME[c]}_embmk'] - df['raw_rbf']
        w = int((gap > 0.005).sum()); l = int((gap < -0.005).sum()); t_ = len(df) - w - l
        print(f'    {NAME[c]:<16} mean gap {gap.mean():+.4f}   win {w:2d} / tie {t_:2d} / lose {l:2d}')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = {**dbg._BASE_CFG, 'lambda_scatter': LAMBDA}
    print(f'device={device}  datasets={len(DATASETS)} (pageblocks excluded)  '
          f'cells={[NAME[c] for c in CELLS]}', flush=True)
    rows, t0 = [], time.time()
    for i, stem in enumerate(DATASETS, 1):
        try:
            rec = run_dataset(stem, device, cfg)
            rows.append(rec)
            print(f'[{i:2d}/{len(DATASETS)}] {stem:<34} raw={rec["raw_rbf"]:.3f}  ' +
                  '  '.join(f'{NAME[c]}={rec[f"{NAME[c]}_embmk"]:.3f}' for c in CELLS), flush=True)
        except Exception as e:
            print(f'[{i:2d}/{len(DATASETS)}] {stem:<34} ERROR {e}', flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)   # incremental save
    df = pd.DataFrame(rows)
    aggregate(df)
    print(f'\nsaved: {OUT}   (elapsed {(time.time()-t0)/60:.1f} min)')
