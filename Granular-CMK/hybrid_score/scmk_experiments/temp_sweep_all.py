"""
temp_sweep_all.py — per-dataset temperature (tau) sweep across all datasets.
================================================================================
The 2x2 factorial showed a single global tau is inadequate (huge per-dataset
variance). Here we sweep tau over a grid on the RAW-bandwidth loss (the minimal
single-knob addition to SCMK, isolating the temperature axis) for every dataset
except pageblocks, seed-2, and map each dataset's tau-response.

Metric: emb-MK precomp AUC ("Embedding + OC-SVM, pre-computed") + SCMK fused, vs
the loss-independent raw-RBF OC-SVM baseline. Reuses run_hybrid_score_semi_debug.

Run:  conda run -n torch311 python temp_sweep_all.py
Saves temp_sweep_results.csv (+ prints the aggregate).
"""
import os, sys, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_hybrid_score_semi_debug as dbg

DIM, LAMBDA, SEED = 64, 100.0, 2
BW_MODE = 'raw'                                  # isolate the temperature axis
TAUS = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]     # 1.0 = original SCMK (no temperature)
DATASETS = [
    'vertebral', 'thyroid', 'wbc_malignant_39_variant1', 'glass', 'ecoli',
    'wine', 'cardio', 'cardiotocography_2and3_33_variant1',
    'tic_tac_toe_negative_12_variant1', 'tic_tac_toe_negative_69_variant1', 'wpbc_variant1',
    'ionosphere_b_24_variant1', 'zoo_variant1', 'sick_sick_72_variant1', 'autos_variant1',
    'annealing_variant1', 'lymphography', 'bands_band_6_variant1', 'audiology_variant1',
]
HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), 'result', 'scmk_experiments')
os.makedirs(RESULT_DIR, exist_ok=True)
OUT = os.path.join(RESULT_DIR, 'temp_sweep_results.csv')
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
    rec = dict(dataset=SHORT.get(stem, stem), raw_rbf=round(raw_rbf, 4))
    for tau in TAUS:
        model, _ = dbg.train_variant(X[tr], dbg.RATIOS, raw_bw, DIM, device, LAMBDA, tau, BW_MODE, cfg, False)
        r = dbg.eval_all(model, X, tr, te, y_te, device)
        rec[f'embmk_t{tau}'] = round(r['emb_mk'], 4)
        rec[f'fused_t{tau}'] = round(r['fused'], 4)
    return rec


def aggregate(df):
    cols = [f'embmk_t{t}' for t in TAUS]
    banner = '=' * 88
    print('\n' + banner)
    print(f'TEMPERATURE SWEEP  (emb-MK precomp AUC; raw bw, dim={DIM}, lam={LAMBDA}, seed={SEED}, {len(df)} datasets)')
    print(banner)

    # per-dataset table + best tau
    hdr = f'  {"dataset":<12}{"raw":>7}' + ''.join(f'{("t"+str(t)):>7}' for t in TAUS) + f'{"best_t":>8}{"gain":>7}'
    print('\n' + hdr)
    best_taus, oracle, orig = [], [], []
    for _, r in df.iterrows():
        vals = [r[c] for c in cols]
        bi = int(np.argmax(vals)); bt = TAUS[bi]
        gain = vals[bi] - r['embmk_t1.0']
        best_taus.append(bt); oracle.append(vals[bi]); orig.append(r['embmk_t1.0'])
        row = f'  {r["dataset"]:<12}{r["raw_rbf"]:>7.3f}' + ''.join(
            (f'{v:>7.3f}' if TAUS[i] != bt else f'{("*%.3f"%v):>7}') for i, v in enumerate(vals))
        print(row + f'{bt:>8}{gain:>+7.3f}')

    print('\n  mean emb-MK by tau (which single global tau is best?):')
    for t in TAUS:
        m = df[f'embmk_t{t}'].mean()
        tag = '  <- original (no temp)' if t == 1.0 else ''
        print(f'    tau={t:<5} mean={m:.4f}{tag}')
    gbt = TAUS[int(np.argmax([df[f"embmk_t{t}"].mean() for t in TAUS]))]
    print(f'  best single global tau = {gbt}  (mean {df[f"embmk_t{gbt}"].mean():.4f})')

    print('\n  per-dataset BEST tau distribution:')
    for t in TAUS:
        n = best_taus.count(t)
        if n:
            print(f'    tau={t:<5} is best for {n:2d} dataset(s)')

    orig_m, orac_m, glob_m, raw_m = np.mean(orig), np.mean(oracle), df[f'embmk_t{gbt}'].mean(), df['raw_rbf'].mean()
    print('\n  SUMMARY (mean emb-MK AUC):')
    print(f'    original (tau=1)          : {orig_m:.4f}')
    print(f'    best single global tau={gbt:<4}: {glob_m:.4f}   ({glob_m-orig_m:+.4f} vs orig)')
    print(f'    per-dataset ORACLE tau    : {orac_m:.4f}   ({orac_m-orig_m:+.4f} vs orig)  [upper bound of tuning]')
    print(f'    raw-RBF baseline          : {raw_m:.4f}')

    # how many datasets does oracle-tau lift above raw, vs original?
    beat_orig = int((np.array(oracle) > np.array(orig) + 0.005).sum())
    beat_raw_o = int((np.array(oracle) > df['raw_rbf'].values + 0.005).sum())
    beat_raw_orig = int((np.array(orig) > df['raw_rbf'].values + 0.005).sum())
    print(f'\n    oracle-tau improves over tau=1 on {beat_orig}/{len(df)} datasets')
    print(f'    emb-MK beats raw-RBF: original {beat_raw_orig}/{len(df)}  ->  oracle-tau {beat_raw_o}/{len(df)}')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = {**dbg._BASE_CFG, 'lambda_scatter': LAMBDA}
    print(f'device={device}  datasets={len(DATASETS)}  bw={BW_MODE}  taus={TAUS}', flush=True)
    rows, t0 = [], time.time()
    for i, stem in enumerate(DATASETS, 1):
        try:
            rec = run_dataset(stem, device, cfg)
            rows.append(rec)
            print(f'[{i:2d}/{len(DATASETS)}] {rec["dataset"]:<12} raw={rec["raw_rbf"]:.3f}  ' +
                  '  '.join(f't{t}={rec[f"embmk_t{t}"]:.3f}' for t in TAUS), flush=True)
        except Exception as e:
            print(f'[{i:2d}/{len(DATASETS)}] {stem} ERROR {e}', flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    aggregate(pd.DataFrame(rows))
    print(f'\nsaved: {OUT}   (elapsed {(time.time()-t0)/60:.1f} min)')
