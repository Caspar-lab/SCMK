"""
build_runtime_table.py — 合并三个计时脚本的结果，输出运行时对比表 + LaTeX。
================================================================================
读入 results/ 下的 scmk_deepod_timing.csv / dfno_timing.csv / disentad_timing.csv，
按数据集对齐，生成：
  results/runtime_merged.csv     —— 每数据集每方法 train_s / infer_s
  runtime_table_train.tex        —— 训练耗时表（datasets × methods + Average）
  runtime_table_infer.tex        —— 推理耗时表
先运行三个计时脚本，再运行本脚本。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
ROOT = r'C:/OD/Shihao/5'

# 方法展示顺序
ORDER = ['SCMK', 'SCMK-1K', 'DeepSVDD', 'Disent-AD', 'NeuTraLAD', 'ICL', 'LMKAD', 'DFNO', 'KFGOD']
DISP = {'SCMK-1K': 'SCMK$_{K=1}$', 'Disent-AD': 'Disent-AD'}

dispname = {'vertebral': 'Vertebral', 'zoo_variant1': 'Zoo', 'wpbc_variant1': 'WPBC',
    'autos_variant1': 'Autos', 'glass': 'Glass', 'audiology_variant1': 'Audiology',
    'bands_band_6_variant1': 'Bands-6', 'annealing_variant1': 'Annealing',
    'cardiotocography_2and3_33_variant1': 'Cardiotoco.', 'thyroid': 'Thyroid',
    'cardio': 'Cardio', 'sick_sick_72_variant1': 'Sick-72', 'lymphography': 'Lympho.',
    'tic_tac_toe_negative_69_variant1': 'TicTacToe-69', 'wine': 'Wine',
    'tic_tac_toe_negative_12_variant1': 'TicTacToe-12', 'ecoli': 'Ecoli',
    'pageblocks_1_258_variant1': 'PageBlocks', 'ionosphere_b_24_variant1': 'Ionosphere',
    'wbc_malignant_39_variant1': 'WBC'}


def load():
    frames = []
    for f in ['scmk_deepod_timing.csv', 'dfno_timing.csv', 'disentad_timing.csv',
              'lmkad_timing.csv', 'kfgod_timing.csv']:
        p = os.path.join(RES, f)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise SystemExit('无计时结果，请先运行 time_*.py')
    return pd.concat(frames, ignore_index=True)


def latex_table(pivot, which, caption, label):
    methods = [m for m in ORDER if m in pivot.columns]
    if which == 'train':   # 直推式方法(DFNO/KFGOD)无训练阶段 → 训练表中剔除全 0 列
        methods = [m for m in methods if not pivot[m].fillna(0).eq(0).all()]
    lines = [r'\begin{table*}[t]', r'\centering', f'\\caption{{{caption}}}',
             f'\\label{{{label}}}', r'\scriptsize', r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabular}{l' + 'c' * len(methods) + '}', r'\toprule',
             'Dataset & ' + ' & '.join(DISP.get(m, m) for m in methods) + r' \\', r'\midrule']
    for stem, row in pivot.iterrows():
        cells = []
        for m in methods:
            v = row[m]
            cells.append('--' if pd.isna(v) else (f'{v:.3f}' if v < 1 else f'{v:.2f}'))
        lines.append(f"{dispname.get(stem, stem):<13}& " + ' & '.join(cells) + r' \\')
    lines.append(r'\midrule')
    avg = ['\\textbf{%s}' % ('%.3f' % pivot[m].mean() if pivot[m].mean() < 1 else '%.2f' % pivot[m].mean())
           for m in methods]
    lines.append(r'\textbf{Average} & ' + ' & '.join(avg) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(lines)


if __name__ == '__main__':
    df = load()
    # 数据集顺序按 D（特征维度）
    sel = pd.read_csv(ROOT + '/result/hybrid_score_semi/selection_scatter_semi_seed2.csv')
    order20 = list(sel['dataset'])
    dord = df.drop_duplicates('dataset').set_index('dataset')['D'].reindex(order20)
    order = list(dord.sort_values().index)

    tr = df.pivot_table(index='dataset', columns='method', values='train_s', aggfunc='first').reindex(order)
    inf = df.pivot_table(index='dataset', columns='method', values='infer_s', aggfunc='first').reindex(order)

    # merged csv
    merged = pd.DataFrame(index=order)
    for m in ORDER:
        if m in tr.columns: merged[f'{m}_train_s'] = tr[m]
        if m in inf.columns: merged[f'{m}_infer_s'] = inf[m]
    merged.insert(0, 'n_test', df.drop_duplicates('dataset').set_index('dataset')['n_test'].reindex(order))
    merged.insert(1, 'D', dord.reindex(order))
    merged.to_csv(os.path.join(RES, 'runtime_merged.csv'))

    open(os.path.join(HERE, 'runtime_table_train.tex'), 'w', encoding='utf-8').write(
        latex_table(tr, 'train',
                    'Training time (seconds) on the seed-2 split (single configuration, same GPU). '
                    'SCMK$_{K=1}$ is the single-kernel variant; DFNO is transductive (no training phase). '
                    'Lower is better.', 'tab:runtime-train'))
    open(os.path.join(HERE, 'runtime_table_infer.tex'), 'w', encoding='utf-8').write(
        latex_table(inf, 'infer',
                    'Inference time (seconds) on the seed-2 test set (median of repeats). '
                    'For DFNO the value is the full transductive computation.', 'tab:runtime-infer'))

    print('=== Training time (s) averages ===')
    for m in ORDER:
        if m in tr.columns: print(f'  {m:<12} {tr[m].mean():.3f}')
    print('=== Inference time (s) averages ===')
    for m in ORDER:
        if m in inf.columns: print(f'  {m:<12} {inf[m].mean():.4f}')
    print('\nsaved: results/runtime_merged.csv, runtime_table_train.tex, runtime_table_infer.tex')
