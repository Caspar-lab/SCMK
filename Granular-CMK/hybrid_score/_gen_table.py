"""Generate the tab:comparison body from master_compare_v2.csv (DFNO now full-data).
Single-valued: KFGOD, DFNO. Mean+/-std: Disent, DeepSVDD, LMKAD, ICL, NeuTraLAD, SCMK.
Bold the cell(s) whose 3-decimal rounded mean equals the row's max rounded mean."""
import pandas as pd
df = pd.read_csv('C:/OD/Shihao/5/result/hybrid_score_semi/master_compare_v2.csv').set_index('stem')

ORDER = ['vertebral', 'thyroid', 'wbc_malignant_39_variant1', 'glass', 'ecoli',
         'pageblocks_1_258_variant1', 'wine', 'cardio', 'cardiotocography_2and3_33_variant1',
         'tic_tac_toe_negative_12_variant1', 'tic_tac_toe_negative_69_variant1', 'wpbc_variant1',
         'ionosphere_b_24_variant1', 'zoo_variant1', 'sick_sick_72_variant1', 'autos_variant1',
         'annealing_variant1', 'lymphography', 'bands_band_6_variant1', 'audiology_variant1']
NAME = {'vertebral': 'Vertebral', 'thyroid': 'Thyroid', 'wbc_malignant_39_variant1': 'WBC',
        'glass': 'Glass', 'ecoli': 'Ecoli', 'pageblocks_1_258_variant1': 'PageBlocks',
        'wine': 'Wine', 'cardio': 'Cardio', 'cardiotocography_2and3_33_variant1': 'Cardiotoco.',
        'tic_tac_toe_negative_12_variant1': 'TicTacToe-12', 'tic_tac_toe_negative_69_variant1': 'TicTacToe-69',
        'wpbc_variant1': 'WPBC', 'ionosphere_b_24_variant1': 'Ionosphere', 'zoo_variant1': 'Zoo',
        'sick_sick_72_variant1': 'Sick-72', 'autos_variant1': 'Autos', 'annealing_variant1': 'Annealing',
        'lymphography': 'Lympho.', 'bands_band_6_variant1': 'Bands-6', 'audiology_variant1': 'Audiology'}
# (key in csv, has_std)
METHODS = [('KFGOD', False), ('Disent', True), ('DeepSVDD', True), ('DFNO', False),
           ('LMKAD', True), ('ICL', True), ('NeuTraLAD', True), ('SCMK', True)]


def cell(mean, std, has_std, bold):
    m = f'{mean:.3f}'
    inner = (f'\\mathbf{{{m}}}' if bold else m)
    if has_std and pd.notna(std):
        return f'${inner}_{{\\pm{std:.3f}}}$'
    return f'${inner}$'


lines = []
means_by_method = {k: [] for k, _ in METHODS}
for st in ORDER:
    r = df.loc[st]
    means = [float(r[f'{k}_mean']) for k, _ in METHODS]
    rmax = max(round(v, 3) for v in means)
    cells = []
    for (k, hs), mv in zip(METHODS, means):
        bold = round(mv, 3) == rmax
        std = r.get(f'{k}_std')
        cells.append(cell(mv, std, hs, bold))
        means_by_method[k].append(mv)
    lines.append(f'{NAME[st]:<12} & ' + ' & '.join(cells) + r' \\')

# Average row: bold the max average (mean of means)
avgs = {k: sum(v) / len(v) for k, v in means_by_method.items()}
amax = max(round(a, 3) for a in avgs.values())
acells = []
for k, hs in METHODS:
    a = avgs[k]
    m = f'{a:.3f}'
    inner = f'\\mathbf{{{m}}}' if round(a, 3) == amax else m
    if hs:
        astd = pd.Series([df.loc[st, f'{k}_std'] for st in ORDER]).astype(float).mean()
        acells.append(f'${inner}_{{\\pm{astd:.3f}}}$')
    else:
        acells.append(f'${inner}$')
avg_line = r'\textbf{Average} & ' + ' & '.join(acells) + r' \\'

print('\n'.join(lines))
print(r'\midrule')
print(avg_line)
print('\n--- per-method averages:', {k: round(a, 4) for k, a in avgs.items()})
