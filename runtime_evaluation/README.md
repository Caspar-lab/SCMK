# SCMK 运行时评估（训练 / 推理耗时）

回应审稿意见：SCMK 需计算 $\binom{K}{2}$ 个跨核亲和矩阵、维护多个投影头，
需给出与基线的**实测训练/推理耗时**对比。本目录提供统一计时脚本，请手动运行。

## 统一口径
- 同一台机器（GPU）、**seed=2 半监督划分**、**单一固定配置**（每方法用其 per-dataset 最优配置）。
- `train_s`：在训练集（仅正常样本）上拟合模型；`infer_s`：对测试集打分。
- CUDA 预热一次后计时；推理取多次重复的中位数。
- 数据集：论文的 20 个（脚本默认）。

## 手动运行（torch311 环境）
```powershell
conda activate torch311
cd "C:\OD\Shihao\5\runtime_evaluation"

# 1) SCMK(K=5) / SCMK(K=1) / DeepSVDD / ICL / NeuTraLAD —— 本进程统一计时
python time_scmk_deepod.py

# 2) DFNO（直推式，无训练阶段；infer 为整体 O(n^2) 计算）
python time_dfno.py

# 3) Disent-AD（复用其官方 repo 的 DisNet；需要 Disent-AD-main 可导入）
python time_disentad.py

# 4) LMKAD（多尺度高斯核单类；复用 LMKAD-master/python）
python time_lmkad.py

# 5) KFGOD（核化模糊-粗糙, 粒球; 直推式; 复用 KFGOD-main/code）
python time_kfgod.py

# 6) 合并成表 + 生成 LaTeX
python build_runtime_table.py

# 7) 出图（条形/折线对比）
python plot_runtime.py
```
加 `--only glass` 可只跑单个数据集做冒烟测试。

## 输出
- `results/scmk_deepod_timing.csv` / `dfno_timing.csv` / `disentad_timing.csv`
- `results/runtime_merged.csv` —— 每数据集每方法 train_s / infer_s
- `runtime_table_train.tex` / `runtime_table_infer.tex` —— 可直接 \input 进 manuscript

## 覆盖范围
| 方法 | 计时方式 | 训练/推理 |
|---|---|---|
| **SCMK** (K=5) | 本仓库代码 | 训练=CMK网络+两路OC-SVM拟合；推理=测试嵌入+打分 |
| **SCMK$_{K=1}$** | 同上，单核 | 直接量化多核 $\binom{K}{2}$ 亲和矩阵的额外开销 |
| DeepSVDD / ICL / NeuTraLAD | DeepOD | fit / decision_function |
| Disent-AD | Disent-AD-main repo | DisNet 训练 / 前向推理 |
| **LMKAD** | LMKAD-master/python | 单类：create_model 训练 / test_model 推理（best_C 取自各 .mat）|
| DFNO | DFNO-main repo | 直推式，仅推理（无训练） |
| **KFGOD** | KFGOD-main/code | 直推式(粒球+核化模糊粗糙)，仅推理（无训练）|

## 冒烟测试（glass, 单数据集）参考
SCMK 训练 3.17s vs SCMK$_{K=1}$ 0.11s（多核开销显著），ICL 6.26s，NeuTraLAD 1.21s，
Disent-AD 1.30s，DeepSVDD 0.28s，DFNO 0.003s（推理，随 n 二次增长）。推理均在毫秒级。
> 说明 SCMK 训练比单核/DeepSVDD 重，但快于 ICL；推理开销可忽略。实际数字以你本机全量运行为准。
