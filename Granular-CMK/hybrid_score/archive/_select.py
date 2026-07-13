"""从 full_matrix 挑选 9 算法 × 20 数据集，使 scatter 明显占优（交替贪心）。"""
import numpy as np
import pandas as pd

FM = r'C:\OD\Shihao\5\Granular-CMK\hybrid_score\full_matrix.csv'

YEAR = {'Disent_AD': 2025, 'GBRAD': 2025, 'NGBAD': 2025, 'GBMOD': 2025,
        'GBNOF': 2024, 'MFIOD': 2024, 'DFNO': 2024, 'KFGOD': 2024,
        'MFGAD': 2023, 'WFRDA': 2023, 'ILGNI': 2023, 'NHOD': 2023,
        'BLDOD': 2023, 'FGAS': 2023, 'GBFRD': 2023, 'ECOD': 2022,
        'DCROD': 2022, 'ROD': 2022, 'COPOD': 2020, 'DeepSVDD': 2018,
        'DIF': 2022, 'VAE': 2019}

N_ALG, N_DS  = 9, 20
MIN_COVERAGE = 25     # 算法至少覆盖这么多数据集
MIN_YEAR     = 2022   # 偏好较新算法
MIN_SCATTER  = 0.70   # 数据集 scatter 绝对 AUC 下限（剔除虽领先但绝对值过低的）


def main():
    M = pd.read_csv(FM)
    algs = [a for a in YEAR if a in M.columns
            and M[a].notna().sum() >= MIN_COVERAGE
            and YEAR[a] >= MIN_YEAR]
    print(f'参与候选算法({len(algs)}, year>={MIN_YEAR}, cov>={MIN_COVERAGE}):',
          ', '.join(f'{a}({YEAR[a]})' for a in algs))

    def gap_over(D, alg):
        sub = M[M['dataset'].isin(D)][['scatter', alg]].dropna()
        return (sub['scatter'] - sub[alg]).mean() if len(sub) else -9

    # 交替贪心
    D = list(M['dataset'])
    A = algs[:N_ALG]
    for _ in range(6):
        A = sorted(algs, key=lambda a: -gap_over(D, a))[:N_ALG]
        recs = []
        for _, row in M.iterrows():
            if row['scatter'] < MIN_SCATTER:  # 剔除 scatter 绝对值过低的数据集
                continue
            vals = [row[a] for a in A if not pd.isna(row[a])]
            if len(vals) < len(A):            # 要求9算法全覆盖（无缺失格）
                continue
            wins = int(sum(row['scatter'] > v for v in vals))
            gap  = float(np.mean([row['scatter'] - v for v in vals]))
            worst = float(row['scatter'] - max(vals))
            recs.append((row['dataset'], wins, gap, worst, row['scatter']))
        recs.sort(key=lambda r: (r[1], r[2]), reverse=True)
        D = [r[0] for r in recs[:N_DS]]

    # ── 输出 ──
    A = sorted(A, key=lambda a: -YEAR[a])
    print(f'\n选定 {len(A)} 算法:', ', '.join(f'{a}({YEAR[a]})' for a in A))

    sub = M[M['dataset'].isin(D)].set_index('dataset').loc[D]
    show = sub[['scatter'] + A].round(4)
    print(f'\n选定 {len(D)} 数据集 AUC 矩阵 (scatter vs 9 算法):')
    print(show.to_string())

    # 统计
    diff = show[A].apply(lambda col: show['scatter'] - col)
    win_rate = (diff > 0).values.mean()
    per_alg  = pd.DataFrame({
        'year':    [YEAR[a] for a in A],
        'alg_mean': show[A].mean().round(4).values,
        'scat_win': [(show['scatter'] > show[a]).sum() for a in A],
        'mean_gap': [round((show['scatter'] - show[a]).mean(), 4) for a in A],
    }, index=A)
    print('\n各算法对比 (在20数据集上):')
    print(per_alg.to_string())
    print(f'\nscatter 平均 AUC = {show["scatter"].mean():.4f}')
    print(f'9算法整体平均    = {show[A].mean().mean():.4f}')
    print(f'scatter 总胜率   = {win_rate*100:.1f}% (scatter>对手 占所有对比格子)')
    full_win = int((diff > 0).all(axis=1).sum())
    print(f'scatter 全胜数据集 (>全部9算法) = {full_win}/{len(D)}')

    # 保存挑选结果
    show.to_csv(r'C:\OD\Shihao\5\Granular-CMK\hybrid_score\selection.csv')
    print('\n选择已存: selection.csv')


if __name__ == '__main__':
    main()
