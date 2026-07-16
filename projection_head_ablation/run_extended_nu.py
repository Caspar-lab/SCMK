"""
run_extended_nu.py — Matched extended-nu sweep.
===============================================
Motivation: under the original nu grid [0.01,0.05,0.1,0.2] the precomputed learned-
kernel variant (Emb-MK precomp) tied the plain raw-RBF OC-SVM (Direct-OCSVM). A
parameter diagnostic showed the tie is partly a grid artifact: on the hard datasets
(Cardio, WBC, Glass) Emb-MK precomp's best AUC is pinned at the nu=0.2 boundary with
AUC still rising, while Direct-OCSVM's optimum is always interior. So the combined
kernel is nu-hungrier and the grid ceiling penalizes it specifically.

This sweep recomputes every nu-dependent variant under BOTH grids, with the SAME
(extended) grid applied to all of them, so the Emb-precomp vs Direct-OCSVM comparison
is fair:

    NU_ORIG = [0.01, 0.05, 0.1, 0.2]                 (as originally reported)
    NU_EXT  = NU_ORIG + [0.3, 0.5, 0.7]              (matched, extended)

Variants (all on the same 50/50 splits, seeds {0,1,2}):
    Raw-MK          raw features         + gauss_med_kernels + dual head
    Direct-OCSVM    raw features         + plain RBF OC-SVM
    Emb-MK (dual)   learned embeddings Z + gauss_med_kernels + dual head
    Emb-MK (precomp)learned embeddings Z + gauss_med_kernels combined -> precomputed OC-SVM
Full is the proposed model as reported (master_compare_v2.csv), shown as reference.

Run (torch311):  conda run -n torch311 python run_extended_nu.py
Outputs (projection_head_ablation/results/):
    extended_nu.csv        per-dataset orig vs ext means/stds + selected nu + deltas
    extended_nu_table.tex  LaTeX table (ext grid), best mean per row bold
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, pandas as pd
import torch  # before sklearn (Windows OpenMP load order)

HERE = os.path.dirname(os.path.abspath(__file__))
HS   = r'C:/OD/Shihao/5/Granular-CMK/hybrid_score'
sys.path.insert(0, HERE); sys.path.insert(0, HS)
from run_hybrid_score_semi import (split_indices, ensemble_auc_split,
                                   gauss_med_kernels, load_data)
from emb_mk import learned_Z, FULL, NAME, ORDER, CFG, DATA
from raw_mk import raw_mk_representation
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, pairwise_distances

OUT   = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
NU_ORIG = [0.01, 0.05, 0.1, 0.2]
NU_EXT  = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]     # matched grid applied to all variants


def best_over(pernu, subset):
    """max AUC over a nu subset; ignore NaNs."""
    vals = [pernu[n] for n in subset if not np.isnan(pernu[n])]
    return max(vals) if vals else float('nan')


def argbest(pernu, subset):
    sub = {n: pernu[n] for n in subset if not np.isnan(pernu[n])}
    return max(sub, key=sub.get) if sub else float('nan')


def direct_rbf_pernu(Xtr, Xte, yte):
    """Plain RBF OC-SVM on raw features, AUC per nu (gamma='scale')."""
    out = {}
    for nu in NU_EXT:
        try:
            out[nu] = roc_auc_score(yte, -OneClassSVM(kernel='rbf', nu=nu).fit(Xtr).decision_function(Xte))
        except Exception:
            out[nu] = float('nan')
    return out


def precomp_pernu(Z, tr, te, yt, kernels):
    """Averaged Gaussian combined kernel on Z -> precomputed OC-SVM, AUC per nu."""
    d2   = pairwise_distances(Z, Z[tr], metric='sqeuclidean')
    comb = np.mean([np.exp(-d2 / (2.0 * k[2]['t'] ** 2)) for k in kernels], axis=0)
    Kfit, Kscr = comb[tr], comb[te]
    out = {}
    for nu in NU_EXT:
        try:
            out[nu] = roc_auc_score(yt, -OneClassSVM(kernel='precomputed', nu=nu).fit(Kfit).decision_function(Kscr))
        except Exception:
            out[nu] = float('nan')
    return out


def agg(seed_vals):
    a = np.array(seed_vals, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1)) if np.sum(~np.isnan(a)) > 1 else 0.0


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device} | seeds={SEEDS} | NU_ORIG={NU_ORIG} | NU_EXT={NU_EXT}', flush=True)
    rows = []
    for stem in ORDER:
        X, y, _ = load_data(os.path.join(DATA, stem + '.mat'))
        yb = (y != 0).astype(int)
        dim, lam = CFG[stem]
        acc = {k: [] for k in ('raw_o', 'raw_e', 'dir_o', 'dir_e',
                               'ed_o', 'ed_e', 'ep_o', 'ep_e', 'ep_nu', 'dir_nu')}
        for sd in SEEDS:
            tr, te = split_indices(yb, sd); yt = yb[te]
            # raw-feature variants
            kX = gauss_med_kernels(X[tr])
            Hp, Hn = raw_mk_representation(X, tr, kX)
            acc['raw_o'].append(ensemble_auc_split(Hp, Hn, tr, te, yt, NU_ORIG))
            acc['raw_e'].append(ensemble_auc_split(Hp, Hn, tr, te, yt, NU_EXT))
            dpn = direct_rbf_pernu(X[tr], X[te], yt)
            acc['dir_o'].append(best_over(dpn, NU_ORIG)); acc['dir_e'].append(best_over(dpn, NU_EXT))
            acc['dir_nu'].append(argbest(dpn, NU_EXT))
            # learned-embedding variants (train once, reuse)
            Z  = learned_Z(X, tr, dim, lam)
            kZ = gauss_med_kernels(Z[tr])
            Hpz, Hnz = raw_mk_representation(Z, tr, kZ)
            acc['ed_o'].append(ensemble_auc_split(Hpz, Hnz, tr, te, yt, NU_ORIG))
            acc['ed_e'].append(ensemble_auc_split(Hpz, Hnz, tr, te, yt, NU_EXT))
            ppn = precomp_pernu(Z, tr, te, yt, kZ)
            acc['ep_o'].append(best_over(ppn, NU_ORIG)); acc['ep_e'].append(best_over(ppn, NU_EXT))
            acc['ep_nu'].append(argbest(ppn, NU_EXT))

        fm, fs = FULL[stem]
        rec = dict(dataset=stem, name=NAME[stem], Full_mean=fm, Full_std=fs)
        for tag, key in [('RawMK', 'raw'), ('Direct', 'dir'), ('EmbDual', 'ed'), ('EmbPrecomp', 'ep')]:
            mo, so = agg(acc[f'{key}_o']); me, se = agg(acc[f'{key}_e'])
            rec[f'{tag}_orig_mean'], rec[f'{tag}_orig_std'] = mo, so
            rec[f'{tag}_ext_mean'],  rec[f'{tag}_ext_std']  = me, se
        rec['EmbPrecomp_ext_bestnu'] = float(np.nanmean(acc['ep_nu']))
        rec['Direct_ext_bestnu']     = float(np.nanmean(acc['dir_nu']))
        rec['d_precompExt_vs_directExt'] = rec['EmbPrecomp_ext_mean'] - rec['Direct_ext_mean']
        rec['d_precomp_ext_vs_orig']     = rec['EmbPrecomp_ext_mean'] - rec['EmbPrecomp_orig_mean']
        rows.append(rec)
        print(f"{NAME[stem]:<13} Full={fm:.4f}  Direct(ext)={rec['Direct_ext_mean']:.4f}  "
              f"EmbPrecomp: orig={rec['EmbPrecomp_orig_mean']:.4f}->ext={rec['EmbPrecomp_ext_mean']:.4f} "
              f"(bestnu~{rec['EmbPrecomp_ext_bestnu']:.2f})  "
              f"vsDirect={rec['d_precompExt_vs_directExt']:+.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'extended_nu.csv'), index=False)

    # ---- LaTeX table (extended grid; Full reference; bold best mean per row) ----
    COLS = [('Full', 'Full_mean', 'Full_std'),
            ('Raw-MK', 'RawMK_ext_mean', 'RawMK_ext_std'),
            ('Direct-OCSVM', 'Direct_ext_mean', 'Direct_ext_std'),
            ('Emb-MK(dual)', 'EmbDual_ext_mean', 'EmbDual_ext_std'),
            ('Emb-MK(precomp)', 'EmbPrecomp_ext_mean', 'EmbPrecomp_ext_std')]
    def fmt(means, stds):
        mx = max(round(m, 3) for m in means)
        return [(f'$\\mathbf{{{m:.3f}}}_{{\\pm{s:.3f}}}$' if round(m, 3) == mx
                 else f'${m:.3f}_{{\\pm{s:.3f}}}$') for m, s in zip(means, stds)]
    lines = []
    for _, r in df.iterrows():
        c = fmt([r[m] for _, m, _ in COLS], [r[s] for _, _, s in COLS])
        lines.append(f"{r['name']:<13} & " + ' & '.join(c) + r' \\')
    amean = [df[m].mean() for _, m, _ in COLS]
    astd  = [df[s].mean() for _, _, s in COLS]
    avg = r'\textbf{Average} & ' + ' & '.join(fmt(amean, astd)) + r' \\'
    header = ' & '.join(n for n, _, _ in COLS)
    tex = (r"\begin{table}[t]\centering" + "\n"
           r"\caption{Extended-$\nu$ sweep (test AUC, mean$_{\pm\text{std}}$ over seeds $0,1,2$; "
           r"matched grid $\nu\in\{0.01,0.05,0.1,0.2,0.3,0.5,0.7\}$ for all variants). "
           r"\emph{Emb-MK} applies the gauss\_med kernels to the learned embeddings; "
           r"\emph{Direct-OCSVM} is a plain RBF OC-SVM on raw features. Best mean per row bold.}" + "\n"
           r"\label{tab:extnu}\small\setlength{\tabcolsep}{5pt}" + "\n"
           r"\begin{tabular}{lccccc}\toprule" + "\n"
           f"Dataset & {header}\\\\\n\\midrule\n" + '\n'.join(lines) +
           "\n\\midrule\n" + avg + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(OUT, 'extended_nu_table.tex'), 'w', encoding='utf-8').write(tex)

    print('\n=== mean AUC over 20 datasets ===')
    for tag, m in [('Full', 'Full_mean'), ('Raw-MK ext', 'RawMK_ext_mean'),
                   ('Direct ext', 'Direct_ext_mean'), ('Emb-dual ext', 'EmbDual_ext_mean'),
                   ('Emb-precomp orig', 'EmbPrecomp_orig_mean'), ('Emb-precomp ext', 'EmbPrecomp_ext_mean')]:
        print(f'  {tag:<18} {df[m].mean():.4f}')
    print(f"\n  Emb-precomp ext - Direct ext : {df['EmbPrecomp_ext_mean'].mean() - df['Direct_ext_mean'].mean():+.4f}")
    print(f"  Emb-precomp ext > Direct ext : {int((df['EmbPrecomp_ext_mean'] > df['Direct_ext_mean']).sum())}/{len(df)}")
    print(f"  Emb-precomp lift orig->ext   : {df['d_precomp_ext_vs_orig'].mean():+.4f}")
    print(f"  mean selected nu (Emb-precomp ext): {df['EmbPrecomp_ext_bestnu'].mean():.3f}  "
          f"(Direct ext: {df['Direct_ext_bestnu'].mean():.3f})")
    print('\nsaved: results/extended_nu.csv, results/extended_nu_table.tex')
