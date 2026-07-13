"""
plot_runtime.py — 由 results/ 的计时结果生成运行时对比图。
输出（本目录）：
  fig_runtime_summary.*   平均 训练/推理 耗时 条形图（方法对比，log 轴）
  fig_train_by_dataset.*  逐数据集 训练耗时 分组条形图（20 数据集 × 各算法）
  fig_runtime_scaling.*   训练/推理 耗时 vs 数据集规模 N（log-log 折线，含标度)
  fig_runtime_all.*       三联组合图
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

_files = ['scmk_deepod_timing.csv', 'dfno_timing.csv', 'disentad_timing.csv',
          'lmkad_timing.csv', 'kfgod_timing.csv']
df = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in _files
                if os.path.exists(os.path.join(RES, f))], ignore_index=True)

disp = {'vertebral': 'Vertebral', 'zoo_variant1': 'Zoo', 'wpbc_variant1': 'WPBC',
    'autos_variant1': 'Autos', 'glass': 'Glass', 'audiology_variant1': 'Audiology',
    'bands_band_6_variant1': 'Bands-6', 'annealing_variant1': 'Annealing',
    'cardiotocography_2and3_33_variant1': 'Cardiotoco.', 'thyroid': 'Thyroid',
    'cardio': 'Cardio', 'sick_sick_72_variant1': 'Sick-72', 'lymphography': 'Lympho.',
    'tic_tac_toe_negative_69_variant1': 'TicTacToe-69', 'wine': 'Wine',
    'tic_tac_toe_negative_12_variant1': 'TicTacToe-12', 'ecoli': 'Ecoli',
    'pageblocks_1_258_variant1': 'PageBlocks', 'ionosphere_b_24_variant1': 'Ionosphere',
    'wbc_malignant_39_variant1': 'WBC'}
LABEL = {'SCMK-1K': r'SCMK$_{K=1}$'}
COLOR = {'SCMK': '#DC143C', 'SCMK-1K': '#FF8C00', 'DeepSVDD': '#20B2AA',
         'Disent-AD': '#BA55D3', 'NeuTraLAD': '#FF69B4', 'ICL': '#8B4513',
         'LMKAD': '#2E8B57', 'DFNO': '#1E90FF', 'KFGOD': '#696969'}
# 有训练阶段（单类/深度）：含 LMKAD；直推式无训练：DFNO、KFGOD
TRAIN_METHODS = ['SCMK', 'SCMK-1K', 'DeepSVDD', 'Disent-AD', 'NeuTraLAD', 'ICL', 'LMKAD']
ALL_METHODS = TRAIN_METHODS + ['DFNO', 'KFGOD']

# 数据集按 n_train 排序
size = df.drop_duplicates('dataset').set_index('dataset')['n_train']
order = list(size.sort_values().index)
tr = df.pivot_table(index='dataset', columns='method', values='train_s', aggfunc='first').reindex(order)
inf = df.pivot_table(index='dataset', columns='method', values='infer_s', aggfunc='first').reindex(order)
lab = lambda m: LABEL.get(m, m)
# 只保留实际有结果的方法（某计时脚本未跑时自动跳过）
TRAIN_METHODS = [m for m in TRAIN_METHODS if m in tr.columns]
ALL_METHODS = [m for m in ALL_METHODS if m in inf.columns]


# ── Fig 1: 平均耗时条形图（train + infer 两面板）──
def barmean(ax, series, methods, title, xlabel):
    vals = [series[m] for m in methods]
    y = np.arange(len(methods))
    bars = ax.barh(y, vals, color=[COLOR[m] for m in methods], edgecolor='black', lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels([lab(m) for m in methods])
    ax.invert_yaxis(); ax.set_xscale('log'); ax.set_xlabel(xlabel); ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3, which='both')
    for b, v in zip(bars, vals):
        ax.text(v * 1.15, b.get_y() + b.get_height() / 2, f'{v:.2f}', va='center', fontsize=9)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
barmean(axes[0], tr[TRAIN_METHODS].mean(), sorted(TRAIN_METHODS, key=lambda m: tr[m].mean()),
        '(a) Mean training time', 'Seconds (log scale)')
barmean(axes[1], inf[ALL_METHODS].mean(), sorted(ALL_METHODS, key=lambda m: inf[m].mean()),
        '(b) Mean inference time', 'Seconds (log scale)')
fig.tight_layout()
for e in ('pdf', 'png'): fig.savefig(os.path.join(HERE, f'fig_runtime_summary.{e}'), bbox_inches='tight', dpi=200)
plt.close(fig)


# ── Fig 2: 逐数据集训练耗时 分组条形图 ──
fig, ax = plt.subplots(figsize=(15, 4.2))
x = np.arange(len(order)); w = 0.8 / len(TRAIN_METHODS)
for i, m in enumerate(TRAIN_METHODS):
    ax.bar(x + i * w - 0.4 + w / 2, tr[m].values, w, label=lab(m),
           color=COLOR[m], edgecolor='black', lw=0.3)
ax.set_yscale('log'); ax.set_ylabel('Training time (s, log)')
ax.set_xticks(x); ax.set_xticklabels([f'{disp[o]}\n(n={int(size[o])})' for o in order],
                                      rotation=45, ha='right', fontsize=8)
ax.set_title('Per-dataset training time (datasets sorted by training-set size $n$)')
ax.legend(ncol=6, loc='upper left', fontsize=9); ax.grid(True, axis='y', alpha=0.3, which='major')
fig.tight_layout()
for e in ('pdf', 'png'): fig.savefig(os.path.join(HERE, f'fig_train_by_dataset.{e}'), bbox_inches='tight', dpi=200)
plt.close(fig)


# ── Fig 3: 耗时 vs 规模 N（log-log 折线）──
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ns = size.reindex(order).values
for m in TRAIN_METHODS:
    o = np.argsort(ns)
    axes[0].plot(ns[o], tr[m].values[o], '-o', color=COLOR[m], label=lab(m), ms=4, lw=1.4)
axes[0].set_xscale('log'); axes[0].set_yscale('log')
axes[0].set_xlabel('Training-set size $n$'); axes[0].set_ylabel('Training time (s)')
axes[0].set_title('(a) Training time vs size'); axes[0].grid(True, alpha=0.3, which='both')
axes[0].legend(fontsize=8, ncol=2)
nt = df.drop_duplicates('dataset').set_index('dataset')['n_test'].reindex(order).values
for m in ALL_METHODS:
    o = np.argsort(nt)
    axes[1].plot(nt[o], inf[m].values[o], '-o', color=COLOR[m], label=lab(m), ms=4, lw=1.4)
axes[1].set_xscale('log'); axes[1].set_yscale('log')
axes[1].set_xlabel('Test-set size'); axes[1].set_ylabel('Inference time (s)')
axes[1].set_title('(b) Inference time vs size'); axes[1].grid(True, alpha=0.3, which='both')
axes[1].legend(fontsize=8, ncol=2)
fig.tight_layout()
for e in ('pdf', 'png'): fig.savefig(os.path.join(HERE, f'fig_runtime_scaling.{e}'), bbox_inches='tight', dpi=200)
plt.close(fig)

print('mean training (s):', {m: round(tr[m].mean(), 3) for m in TRAIN_METHODS})
print('mean inference (s):', {m: round(inf[m].mean(), 4) for m in ALL_METHODS})
print('saved: fig_runtime_summary.*, fig_train_by_dataset.*, fig_runtime_scaling.*')
