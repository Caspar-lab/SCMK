"""
在 BBCSport 双视图数据集上运行 Granular-CMK，遍历 5 种核函数。
数据来源：../CMK-code_release/data/bbcsport_2view.mat
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from GCMK import default_args, main

data_name = 'bbcsport_2view'
data_dir  = '../CMK-code_release/data'

kernels = [
    ('Gaussian',   {'type': 'Gaussian',   't': 1.0}),
    ('Linear',     {'type': 'Linear'}),
    ('Polynomial', {'type': 'Polynomial', 'a': 1.0, 'b': 1.0, 'd': 2.0}),
    ('Sigmoid',    {'type': 'Sigmoid',    'd': 2.0, 'c': 0.0}),
    ('Cauchy',     {'type': 'Cauchy',     'sigma': 1.0}),
]

for name, opts in kernels:
    print(f'\n=== {name} kernel ===')
    args = default_args(
        data_name=data_name,
        normalize=True,
        latent_dim=128,
        learning_rate=1.0,
        epochs=450,          # 150 CMK + 300 CMKKM（与 CMK 对齐）
        view_mode='multiview',
        data_dir=data_dir,
    )
    args.kernel_options = opts
    main(args)

print('\nAll kernels done.')
