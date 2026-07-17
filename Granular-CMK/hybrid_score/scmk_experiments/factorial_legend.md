# Legend: the four factorial versions vs. the original SCMK

There are **two independent axes**. Don't conflate them:
- **Axis 1 — the four "versions"** differ *only in the contrastive training loss*.
- **Axis 2 — the scoring abbreviations** (`raw-RBF`, `emb-lin`, `mag`, `fused`, `emb-MK`)
  are *detectors applied to a trained model*, independent of which version produced it.

The factorial (`factorial_all.py` / `factorial_all_results.csv`) was run at a fixed
`dim=64, λ=100, seed=2`, ν grid-searched, on 19 datasets (the 20 minus PageBlocks).

---

## The original SCMK proposal (paper / `run_hybrid_score_semi.py`)

- **Loss:** cross-kernel InfoNCE with Gaussian bandwidths from the **raw-feature**
  median (`gauss_med_kernels`), **no temperature**, plus `λ·scatter_loss`.
- **Representation:** `K=5` learnable linear projection heads (`CMKNet`).
- **Detector:** **dual-signal max-ensemble** =
  `max( minmax(linear-OCSVM on L2-normalised concat embeddings),
        minmax(RBF-OCSVM on per-kernel projection norms) )`.
- **Protocol:** per-dataset **best** `(dim, λ, ν)`, averaged over **seeds {0,1,2}**,
  on **20** datasets.

---

## Axis 1 — the four versions (change the LOSS only)

| tag in CSV        | kernel bandwidth in the loss        | temperature τ | change vs SCMK loss        |
|-------------------|-------------------------------------|---------------|----------------------------|
| `orig(raw,t1)`    | raw-feature median (fixed)          | 1.0 (none)    | **identical to SCMK's loss** |
| `temp(raw,t.2)`   | raw-feature median (fixed)          | 0.2           | + temperature only         |
| `bw(emb,t1)`      | embedding-distance median (per step)| 1.0 (none)    | + embedding bandwidth only |
| `both(emb,t.2)`   | embedding-distance median (per step)| 0.2           | + both changes             |

Everything else is held identical across all four and to SCMK: `K=5` heads, the
scatter loss `λ=100`, Adam, 100 epochs, L2-normalised embeddings.
So `orig(raw,t1)` is the **control** = SCMK's own loss; the other three are the two
"fixes" and their combination.

- **temperature τ:** `logits = exp(K_avg / τ)`. τ=1 is the original; τ<1 sharpens the
  InfoNCE softmax (more contrast).
- **embedding bandwidth:** recompute each step `t_k = median(embedding pairwise dist) × ratio_k`
  instead of using the raw-feature median (fixes the "dead/saturated kernel" mismatch).

---

## Axis 2 — the scoring detectors (how a trained model → AUC)

| name              | detector                                                        | relation to SCMK                    |
|-------------------|-----------------------------------------------------------------|-------------------------------------|
| `raw-RBF` (`raw_rbf`) | RBF OC-SVM on the **raw features** X                         | baseline; no embedding; **loss-independent** (same for all 4 versions) |
| `emb-lin` (`_emblin`) | linear OC-SVM on the concat L2-normalised embeddings         | **= SCMK's "direction" signal**     |
| `mag`             | RBF OC-SVM on the per-kernel projection norms                    | **= SCMK's "magnitude" signal**     |
| `fused` (`_fused`)| `max(minmax(emb-lin), minmax(mag))`                             | **= the actual SCMK detector**      |
| `emb-MK` (`_embmk`)| pre-computed OC-SVM with an **averaged multi-kernel Gaussian Gram on the embeddings** (embedding-median bandwidths) | a **diagnostic** ("Embedding + OC-SVM, pre-computed"); **NOT part of SCMK** |

---

## Anti-confusion notes

1. **The four versions change training only.** The CSV has three scored columns per
   version — `<tag>_embmk`, `<tag>_emblin`, `<tag>_fused` — i.e. that trained model run
   through three of the Axis-2 detectors. `raw_rbf` is a single constant column.
2. **The factorial aggregate used `emb-MK`**, because that is the
   "embedding + OC-SVM (pre-computed)" you asked about — it is **not** SCMK's `fused`
   detector, and its numbers are **not** comparable to Table 2's SCMK AUC.
3. **`orig(raw,t1)` ≠ Table 2 SCMK**, even though the *loss* is identical, because here it is
   (a) scored by `emb-MK` (not `fused`), (b) at fixed `dim=64, λ=100` (not per-dataset best),
   (c) on a single seed-2 split (not the 3-seed mean), (d) over 19 datasets (not 20).
   Use it as an internal *control*, not as the paper's reported number.
4. **`raw-RBF` is the yardstick** ("standard OC-SVM"): it doesn't depend on the loss, so
   it's the same across all four versions and is what "does embedding+OCSVM beat plain
   OCSVM?" is measured against.

---

## One-line summary of the finding (so the numbers make sense)

Across the 19 datasets, on `emb-MK`: `orig` 0.8405, `temp` 0.8436, `bw` 0.8506,
`both` 0.8218, vs `raw-RBF` 0.8395. → No version is a consistent win; the fixes help a
few "collapsed" datasets (cardio, cardiotoco) but hurt several already-good ones
(audiology, annealing, TicTacToe), netting roughly neutral-to-negative on average.
The single-dataset (cardio) result was **not** representative.


# 四种因子版本对比原始 SCMK

存在**两个独立维度**，请勿混淆：
- **维度一 —— 四种“版本”**：差异**仅在于对比训练损失函数**。
- **维度二 —— 评分缩写**（`raw-RBF`、`emb-lin`、`mag`、`fused`、`emb-MK`）
  是**应用于已训练模型的检测器**，与模型由哪个版本生成无关。

因子实验（`factorial_all.py` / `factorial_all_results.csv`）在固定参数
`dim=64, λ=100, seed=2`、对 ν 进行网格搜索的条件下，于 19 个数据集（共 20 个，剔除 PageBlocks）上运行。

---

## 原始 SCMK 方案（论文 / `run_hybrid_score_semi.py`）

- **损失函数：** 采用高斯带宽源自**原始特征**中位数（`gauss_med_kernels`）的交叉核 InfoNCE，**无温度参数**，外加 `λ·scatter_loss`。
- **表征：** `K=5` 个可学习的线性投影头（`CMKNet`）。
- **检测器：** **双信号最大集成** =
  `max( minmax(基于 L2 归一化拼接嵌入的线性 OCSVM),
        minmax(基于各核投影范数的 RBF-OCSVM) )`。
- **协议：** 每个数据集选取**最优** `(dim, λ, ν)`，在 **20** 个数据集上取 **{0,1,2} 三个随机种子**的平均值。

---

## 维度一 —— 四种版本（仅修改损失函数）

| CSV 中的标签     | 损失函数中的核带宽                 | 温度 τ    | 相对于 SCMK 损失的改动       |
|------------------|------------------------------------|-----------|------------------------------|
| `orig(raw,t1)`   | 原始特征中位数（固定）             | 1.0（无） | **与 SCMK 损失完全一致**     |
| `temp(raw,t.2)`  | 原始特征中位数（固定）             | 0.2       | 仅增加温度参数               |
| `bw(emb,t1)`     | 嵌入距离中位数（每步更新）         | 1.0（无） | 仅增加嵌入带宽               |
| `both(emb,t.2)`  | 嵌入距离中位数（每步更新）         | 0.2       | 同时增加上述两项改动         |

其余所有设置在这四种版本及 SCMK 中均保持一致：`K=5` 个投影头、
散度损失 `λ=100`、Adam 优化器、100 轮训练、L2 归一化嵌入。
因此，`orig(raw,t1)` 是**对照组** = SCMK 自身的损失；其余三个版本分别对应两种
“修正”及其组合。

- **温度 τ：** `logits = exp(K_avg / τ)`。τ=1 为原始设定；τ<1 会使 InfoNCE 的 Softmax 分布更尖锐（对比度更强）。
- **嵌入带宽：** 每一步重新计算 `t_k = median(嵌入两两距离) × ratio_k`，
  而非使用原始特征中位数（解决“死核/饱和核”不匹配问题）。

---

## 维度二 —— 评分检测器（如何将训练好的模型转化为 AUC）

| 名称              | 检测器                                                            | 与 SCMK 的关系                      |
|-------------------|-------------------------------------------------------------------|-------------------------------------|
| `raw-RBF` (`raw_rbf`) | 基于**原始特征** X 的 RBF OC-SVM                                | 基线；无嵌入；**与损失无关**（四版本通用） |
| `emb-lin` (`_emblin`) | 基于拼接后 L2 归一化嵌入的线性 OC-SVM                          | **= SCMK 的“方向”信号**             |
| `mag`             | 基于各核投影范数的 RBF OC-SVM                                      | **= SCMK 的“幅度”信号**             |
| `fused` (`_fused`)| `max(minmax(emb-lin), minmax(mag))`                               | **= 实际的 SCMK 检测器**            |
| `emb-MK` (`_embmk`)| 预计算的 OC-SVM，使用基于嵌入的**平均多核高斯 Gram 矩阵**（嵌入中位数带宽） | 一种**诊断性指标**（“嵌入 + OC-SVM，预计算”）；**不属于 SCMK** |

---

## 防混淆说明

1. **四种版本仅改变训练过程。** CSV 中每个版本对应三列评分 —— `<tag>_embmk`、`<tag>_emblin`、`<tag>_fused` —— 即该训练模型通过维度二的三种检测器得到的结果。`raw_rbf` 是单一常量列。
2. **因子汇总使用了 `emb-MK`**，因为这就是你询问的
   “嵌入 + OC-SVM（预计算）”方法 —— 它**不是** SCMK 的 `fused` 检测器，其数值**不可**与表 2 中的 SCMK AUC 直接比较。
3. **`orig(raw,t1)` ≠ 表 2 中的 SCMK**，尽管*损失函数*相同，但此处：(a) 使用 `emb-MK` 评分（非 `fused`），(b) 固定 `dim=64, λ=100`（非逐数据集最优），(c) 单种子 seed-2 划分（非 3 种子均值），(d) 覆盖 19 个数据集（非 20 个）。请将其用作内部*对照*，而非论文中报告的数值。
4. **`raw-RBF` 是基准线**（“标准 OC-SVM”）：它不依赖于损失函数，因此在四个版本中数值相同，用于衡量“嵌入+OCSVM 是否优于纯 OCSVM”。

---

## 核心发现一句话总结（助你理解数据）

在 19 个数据集上，针对 `emb-MK` 的结果：`orig` 0.8405、`temp` 0.8436、`bw` 0.8506、
`both` 0.8218，对比 `raw-RBF` 0.8395。→ 没有任何版本取得全面胜利；修正措施帮助了少数“崩溃”的数据集（如 cardio、cardiotoco），但在多个原本表现良好的数据集（如 audiology、annealing、TicTacToe）上造成了损害，平均而言大致呈中性至负面效果。
单一数据集（cardio）的结果**不具备代表性**。