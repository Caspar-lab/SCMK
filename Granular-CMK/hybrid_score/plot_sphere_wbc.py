"""
plot_sphere_wbc.py — wbc(幅值主导)球面 3D 图
============================================
直接运行(VSCode Run / python plot_sphere_wbc.py)会弹出交互窗口:
  · 鼠标左键拖动 = 旋转到任意角度
  · 满意后点窗口工具栏的"保存"(磁盘图标)导出 PDF/PNG
保持默认交互后端(不要 Agg)才能弹窗。
"""
import os, sys
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt          # 默认交互后端 -> plt.show() 弹窗
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_hybrid_score import load_data, gauss_med_kernels, extract_components
from CMK_OCSVM import TRAIN_CFG
from CMK_OCSVM_scatter import train_cmk_scatter

# ════════════════════════════ 配置 ════════════════════════════
STEM      = 'wbc_malignant_39_variant1'
TITLE     = 'WBC  (magnitude-dominant)'
DATA_ROOT = r'C:\OD\Shihao\datasets'
# ══════════════════════════════════════════════════════════════

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
best = pd.read_csv(r'C:\OD\Shihao\5\result\hybrid_score\hybrid_best.csv')
CFG = {r.dataset: (int(r.best_dim), float(r.best_lambda)) for r in best.itertuples()}
dim, lam = CFG[STEM]

X, y, _ = load_data(os.path.join(DATA_ROOT, STEM + '.mat'))
ker = gauss_med_kernels(X[y == 0])
m = train_cmk_scatter(X, y, ker, dim, dev, {**TRAIN_CFG, 'lambda_scatter': lam})
Hn, Hnorms = extract_components(m, X, dev)
Hdir = np.concatenate(Hn, 1)
no, an = y == 0, y == 1

# 方向：判别投影到单位球面 —— 主轴 = "异常-正常均值差"(最分离方向)，另两轴为正交补 PCA。
# 注：此投影仅用于可视化、揭示判别方向(用到标签)；训练与评分本身不使用任何标签。
# wbc 方向本不可分(s_dir≈0.57)，故方向轴上正常/异常仍混叠，分离靠半径(幅值)。
w = Hdir[an].mean(0) - Hdir[no].mean(0); w = w / (np.linalg.norm(w) + 1e-9)
a1 = Hdir @ w
a23 = PCA(2).fit_transform(Hdir - np.outer(a1, w))
u3 = np.column_stack([a1, a23])
u3 = u3 / (np.linalg.norm(u3, axis=1, keepdims=True) + 1e-9)

# 幅值：对数压缩半径(正常≈1 贴球面，异常鼓出)
z = Hnorms.mean(1)
rn = 1.0 + np.log(z / (np.median(z[y == 0]) + 1e-9) + 1e-9)
rn = np.clip(rn, 0.2, None)
P = rn[:, None] * u3

uu = np.linspace(0, 2 * np.pi, 48); vv = np.linspace(0, np.pi, 24)
SX = np.outer(np.cos(uu), np.sin(vv))
SY = np.outer(np.sin(uu), np.sin(vv))
SZ = np.outer(np.ones_like(uu), np.cos(vv))

fig = plt.figure(figsize=(7.5, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(SX, SY, SZ, color='gray', alpha=0.10, linewidth=0, shade=False)
ax.plot_wireframe(SX, SY, SZ, color='gray', alpha=0.18, linewidth=0.3)
ax.scatter(P[no, 0], P[no, 1], P[no, 2], c='steelblue', s=14, alpha=0.55,
           edgecolors='none', label='normal')
ax.scatter(P[an, 0], P[an, 1], P[an, 2], c='crimson', s=52, marker='^',
           edgecolors='k', linewidths=0.3, label='anomaly')
rmax = max(1.2, rn[an].max() * 1.05)
ax.set_xlim(-rmax, rmax); ax.set_ylim(-rmax, rmax); ax.set_zlim(-rmax, rmax)
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('discriminative dir'); ax.set_ylabel('dir-2'); ax.set_zlabel('dir-3')
ax.set_title(f'{TITLE}\n(sphere = unit-norm direction, radius = 1+log magnitude)',
             fontsize=12)

# 初始视角自动朝异常质心(最能看出聚集)；弹窗后可继续鼠标拖动微调
_c = P[an].mean(0); _c = _c / (np.linalg.norm(_c) + 1e-9)
ax.view_init(elev=float(np.degrees(np.arcsin(np.clip(_c[2], -1, 1)))),
             azim=float(np.degrees(np.arctan2(_c[1], _c[0]))))
ax.legend(loc='upper left', fontsize=11)
fig.tight_layout()

print(f'[{STEM}] dim={dim} lam={lam}  正常={int(no.sum())} 异常={int(an.sum())}')
print('鼠标左键拖动旋转;满意后用窗口工具栏的保存按钮(磁盘图标)导出 PDF。')
plt.show()
