"""Rewrite plot_roc_compare.ipynb to the corrected 8-method generator (DFNO transductive
on the full dataset, consistent with KFGOD). Mirrors regen_roc_dfno_fulldata.py."""
import json, os

NB = 'C:/OD/Shihao/5/Granular-CMK/hybrid_score/plot_roc_compare.ipynb'


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


c_intro = md(
    "# ROC comparison: SCMK vs 7 recent detectors (semi 50/50, seed=2)\n"
    "\n"
    "One panel per dataset overlays the 8 methods' ROC curves. 20 datasets from\n"
    "`selection_scatter_semi_seed2.csv`.\n"
    "\n"
    "**Protocol (per method):**\n"
    "- `SCMK`: seed-2 semi split, live CMK training, scored on the held-out test set.\n"
    "- `KFGOD`, `DFNO`: transductive, scored on the FULL dataset (opt_out_scores) — DFNO\n"
    "  uses `Experimental_results/DFNO_results`.\n"
    "- `Disent-AD`/`DeepSVDD`/`LMKAD`: seed-2 split, cached per-sample test scores (res_single).\n"
    "- `ICL`/`NeuTraLAD`: seed-2 split, cached per-sample test scores (CSV).")

c_imports = code(
    "import os, sys\n"
    "os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import scipy.io as sio\n"
    "import torch\n"
    "from sklearn.metrics import roc_curve, roc_auc_score\n"
    "\n"
    "NB_DIR = 'C:/OD/Shihao/5/Granular-CMK/hybrid_score'\n"
    "if NB_DIR not in sys.path:\n"
    "    sys.path.insert(0, NB_DIR)\n"
    "# self-contained semi module (does NOT import matplotlib -> avoids the Windows\n"
    "# OpenMP crash that importing the old run_hybrid_score triggers alongside torch)\n"
    "from run_hybrid_score_semi import (split_indices, extract_components, _best_ocsvm_scores,\n"
    "                                   _minmax, gauss_med_kernels, load_data, _BASE_CFG,\n"
    "                                   NU_CANDIDATES)\n"
    "from CMK_OCSVM_scatter import train_cmk_scatter\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "DATA_ROOT  = 'C:/OD/Shihao/datasets'\n"
    "ER         = 'C:/OD/Shihao/Experimental_results'\n"
    "KFGOD_DIR  = 'C:/OD/Shihao/KFGOD-main/results'\n"
    "DFNO_DIR   = ER + '/DFNO_results'                 # full-data transductive DFNO\n"
    "ICL_DIR    = 'C:/OD/Shihao/5/ICL and NeuTraLAD/results_split/scores/seed2'\n"
    "SEMI_DIR   = 'C:/OD/Shihao/5/result/hybrid_score_semi'\n"
    "RESULT_DIR = SEMI_DIR\n"
    "SEL_CSV    = os.path.join(SEMI_DIR, 'selection_scatter_semi_seed2.csv')\n"
    "ALL_CSV    = os.path.join(SEMI_DIR, 'hybrid_semi_all.csv')\n"
    "SPLIT_SEED = 2\n"
    "plt.rcParams['font.size'] = 13\n"
    "plt.rcParams['font.family'] = 'Times New Roman'\n"
    "device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')\n"
    "print('device:', device, '| split_seed:', SPLIT_SEED)")

c_methods = code(
    "# Current 8-method set (plot order). DFNO is transductive on the FULL dataset, like KFGOD.\n"
    "METHODS = ['KFGOD', 'Disent_AD', 'DeepSVDD', 'DFNO', 'LMKAD', 'ICL', 'NeuTraLAD', 'scatter']\n"
    "LABEL = {'KFGOD': 'KFGOD', 'Disent_AD': 'Disent-AD', 'DeepSVDD': 'DeepSVDD', 'DFNO': 'DFNO',\n"
    "         'LMKAD': 'LMKAD', 'ICL': 'ICL', 'NeuTraLAD': 'NeuTraLAD', 'scatter': 'SCMK'}\n"
    "STYLE = {'KFGOD': ('#FF8C00', 'x', 1.1), 'Disent_AD': ('#BA55D3', '^', 1.1),\n"
    "         'DeepSVDD': ('#20B2AA', '<', 1.1), 'DFNO': ('#00BFFF', 'v', 1.1),\n"
    "         'LMKAD': ('#6B8E23', '>', 1.1), 'ICL': ('#8B4513', 's', 1.1),\n"
    "         'NeuTraLAD': ('#FF69B4', 'p', 1.1), 'scatter': ('#DC143C', 'o', 1.5)}\n"
    "\n"
    "sel = pd.read_csv(SEL_CSV); DATASETS = list(sel['dataset'])\n"
    "_all = pd.read_csv(ALL_CSV); _s2 = _all[_all['split_seed'] == SPLIT_SEED]\n"
    "BEST_CFG = {}\n"
    "for st in DATASETS:\n"
    "    sub = _s2[_s2['dataset'] == st]; br = sub.loc[sub['auc'].idxmax()]\n"
    "    BEST_CFG[st] = (int(br['latent_dim']), float(br['lambda_scatter']))\n"
    "print(f'{len(DATASETS)} datasets | 7 comparison algs + SCMK | DFNO=full-data transductive')")

c_loaders = code(
    "def load_full_y(stem):\n"
    "    t = sio.loadmat(os.path.join(DATA_ROOT, stem + '.mat'))['trandata'].astype(float)\n"
    "    return (t[:, -1] != 0).astype(int)\n"
    "\n"
    "def full_data_scores(path, y):\n"
    "    if not os.path.exists(path):\n"
    "        return None\n"
    "    s = np.asarray(sio.loadmat(path)['opt_out_scores'])[:, 0].ravel()\n"
    "    return s if len(s) == len(y) else None\n"
    "\n"
    "def seed2_mat_curve(path):\n"
    "    r = sio.loadmat(path)['res_single'][0, 0]\n"
    "    return (np.asarray(r['labels'], float).ravel().astype(int),\n"
    "            np.asarray(r['opt_scores'], float).ravel())\n"
    "\n"
    "def seed2_csv_curve(path):\n"
    "    d = pd.read_csv(path)\n"
    "    return d['label'].values.astype(int), d['anomaly_score'].values.astype(float)\n"
    "\n"
    "def scatter_test_scores(stem, train_idx, test_idx):\n"
    "    X, y, _ = load_data(os.path.join(DATA_ROOT, stem + '.mat'))\n"
    "    dim, lam = BEST_CFG[stem]\n"
    "    kernels = gauss_med_kernels(X[train_idx])\n"
    "    cfg = {**_BASE_CFG, 'lambda_scatter': lam}\n"
    "    model = train_cmk_scatter(X[train_idx], np.zeros(len(train_idx), int), kernels, dim, device, cfg)\n"
    "    H_norm_per, H_norms = extract_components(model, X, device)\n"
    "    H_dir = np.concatenate(H_norm_per, axis=1)\n"
    "    y_test = (y[test_idx] != 0).astype(int)\n"
    "    _, _, s_dir = _best_ocsvm_scores(H_dir[test_idx], y_test, H_dir[train_idx], 'linear', NU_CANDIDATES)\n"
    "    _, _, s_nrm = _best_ocsvm_scores(H_norms[test_idx], y_test, H_norms[train_idx], 'rbf', NU_CANDIDATES)\n"
    "    if s_dir is None: return y_test, _minmax(s_nrm)\n"
    "    if s_nrm is None: return y_test, _minmax(s_dir)\n"
    "    return y_test, np.maximum(_minmax(s_dir), _minmax(s_nrm))\n"
    "\n"
    "def method_curve(m, stem, y_full, train_idx, test_idx):\n"
    "    if m == 'scatter':\n"
    "        return scatter_test_scores(stem, train_idx, test_idx)\n"
    "    if m == 'KFGOD':\n"
    "        s = full_data_scores(os.path.join(KFGOD_DIR, stem, f'{stem}_KFGOD.mat'), y_full)\n"
    "        return (y_full, s) if s is not None else None\n"
    "    if m == 'DFNO':\n"
    "        s = full_data_scores(os.path.join(DFNO_DIR, stem, f'{stem}_DFNO.mat'), y_full)\n"
    "        return (y_full, s) if s is not None else None\n"
    "    if m == 'Disent_AD':\n"
    "        return seed2_mat_curve(os.path.join(ER, 'Disent_AD_split_seed2', f'{stem}_DisentAD.mat'))\n"
    "    if m == 'DeepSVDD':\n"
    "        return seed2_mat_curve(os.path.join(ER, 'DeepSVDD_split_seed2', f'{stem}_DeepSVDD.mat'))\n"
    "    if m == 'LMKAD':\n"
    "        return seed2_mat_curve(os.path.join(ER, 'LMKAD_gauss_split_seed2', f'{stem}_LMKAD.mat'))\n"
    "    if m == 'ICL':\n"
    "        return seed2_csv_curve(os.path.join(ICL_DIR, 'ICL', f'{stem}_scores.csv'))\n"
    "    if m == 'NeuTraLAD':\n"
    "        return seed2_csv_curve(os.path.join(ICL_DIR, 'NeuTraL', f'{stem}_scores.csv'))\n"
    "    return None")

c_compute = code(
    "# Compute all ROC curves. RESULTS[stem][method] = (fpr, tpr, auc)\n"
    "RESULTS = {}\n"
    "for stem in DATASETS:\n"
    "    y_full = load_full_y(stem)\n"
    "    train_idx, test_idx = split_indices(y_full, SPLIT_SEED)\n"
    "    RESULTS[stem] = {}\n"
    "    for m in METHODS:\n"
    "        try:\n"
    "            out = method_curve(m, stem, y_full, train_idx, test_idx)\n"
    "            if out is None: continue\n"
    "            y_eval, sc = out\n"
    "            fpr, tpr, _ = roc_curve(y_eval, sc)\n"
    "            RESULTS[stem][m] = (fpr, tpr, roc_auc_score(y_eval, sc))\n"
    "        except Exception as e:\n"
    "            print('ERR', m, stem, ':', e)\n"
    "    print(f'{stem:<36} ' + '  '.join(f'{LABEL[m]}={RESULTS[stem][m][2]:.3f}'\n"
    "                                     for m in METHODS if m in RESULTS[stem]))\n"
    "print('done')")

c_plot_md = md("## Per-dataset ROC panels -> result/hybrid_score_semi/roc_compare/{stem}_ROC.pdf")

c_plot = code(
    "DISP = {'vertebral':'Vertebral','thyroid':'Thyroid','wbc_malignant_39_variant1':'WBC',\n"
    "        'glass':'Glass','ecoli':'Ecoli','pageblocks_1_258_variant1':'PageBlocks','wine':'Wine',\n"
    "        'cardio':'Cardio','cardiotocography_2and3_33_variant1':'Cardiotoco.',\n"
    "        'tic_tac_toe_negative_69_variant1':'TicTacToe-69','tic_tac_toe_negative_12_variant1':'TicTacToe-12',\n"
    "        'wpbc_variant1':'WPBC','ionosphere_b_24_variant1':'Ionosphere','zoo_variant1':'Zoo',\n"
    "        'sick_sick_72_variant1':'Sick-72','autos_variant1':'Autos','annealing_variant1':'Annealing',\n"
    "        'lymphography':'Lympho.','bands_band_6_variant1':'Bands-6','audiology_variant1':'Audiology'}\n"
    "out_dir = os.path.join(RESULT_DIR, 'roc_compare'); os.makedirs(out_dir, exist_ok=True)\n"
    "for stem in DATASETS:\n"
    "    res = RESULTS.get(stem, {})\n"
    "    if not res: continue\n"
    "    fig = plt.figure(figsize=(4, 3), dpi=150)\n"
    "    plt.plot([0, 1], [0, 1], color='gray', lw=0.5, linestyle=(0, (8, 8)))\n"
    "    for m in METHODS:\n"
    "        if m not in res: continue\n"
    "        fpr, tpr, auc = res[m]; col, mk, lw = STYLE[m]\n"
    "        plt.plot(fpr, tpr, label=f'{LABEL[m]} ({auc:.3f})', color=col, marker=mk,\n"
    "                 markevery=max(len(fpr)//8, 1), markersize=3, lw=lw)\n"
    "    plt.xticks([0,0.2,0.4,0.6,0.8,1], [0,20,40,60,80,100], fontsize=7)\n"
    "    plt.yticks([0,0.2,0.4,0.6,0.8,1], [0,20,40,60,80,100], fontsize=7)\n"
    "    plt.xlim(-0.05, 1.02); plt.ylim(-0.05, 1.02); plt.grid(True)\n"
    "    plt.title(DISP.get(stem, stem[:30]), fontsize=9)\n"
    "    plt.legend(prop={'size': 5}, ncol=2, loc='lower right')\n"
    "    fig.patch.set_facecolor('white')\n"
    "    plt.savefig(os.path.join(out_dir, stem + '_ROC.pdf'), bbox_inches='tight', pad_inches=0.02)\n"
    "    plt.show(); plt.close()\n"
    "print('saved per-dataset ROC to:', out_dir)")

nb = {"cells": [c_intro, c_imports, c_methods, c_loaders, c_compute, c_plot_md, c_plot],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

if not os.path.exists(NB + '.bak_oldmethods'):
    os.rename(NB, NB + '.bak_oldmethods') if os.path.exists(NB) else None
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('wrote', NB, '(backup: .bak_oldmethods)')
