# SCMK 方法贡献总结

> 基于 `Granular-CMK` 中 scatter 相关代码归纳。方法由三个文件层层叠加构成,
> 对应三项递进的贡献:跨多核对比表示 → 多核散度紧凑性惩罚 → 方向/幅值双信号融合检测。

```
CMK_OCSVM.py            ─ 跨多核对比表示(底座)
   └─ CMK_OCSVM_scatter.py   ─ + 多核散度紧凑性惩罚(核心创新)
        └─ hybrid_score/run_hybrid_score.py ─ + 方向/幅值双信号检测(max_ensemble)
```

---

## 贡献 1 ─ 跨多核对比表示(CMK 底座)

文件:[CMK_OCSVM.py](CMK_OCSVM.py)

- **多尺度核库**:`gauss_med_kernels`（[CMK_OCSVM.py:136](CMK_OCSVM.py#L136)）以**正常样本**欧氏距离中位数 `med` 生成 5 个高斯核,带宽 `med×{0.1, 0.5, 1, 2, 5}`,从局部到全局覆盖多尺度邻域结构;仅用正常样本估计带宽,避免异常点污染距离统计。
- **K 个独立无偏置线性投影头** `CMKNet`（[CMK_OCSVM.py:152](CMK_OCSVM.py#L152)）:每种核对应一个 `W_k : R^D → R^d`。
  - 无偏置(`bias=False`):保持嵌入以原点为中心,契合 OC-SVM 的超平面假设。
  - L2 归一化:将各核嵌入约束到单位超球面,统一各核尺度。
- **跨核 InfoNCE** `cross_kernel_loss`（[CMK_OCSVM.py:205](CMK_OCSVM.py#L205)）:对所有 C(K,2) 个核对 (k, l),把"同一样本 i 在核 k、核 l 下的两个嵌入"作为正对,不同样本为负对,强制**跨核一致性**。仅用正常样本(`y==0`)训练,异常样本因偏离训练分布而无法与正常簇对齐。

**局限(贡献 2 的动机):** InfoNCE 只保证"跨核一致",并不强制正常样本聚成**紧致簇**,OC-SVM 面对的正常嵌入散布范围大、边界松弛。

---

## 贡献 2 ─ 多核散度紧凑性惩罚(核心创新)

文件:[CMK_OCSVM_scatter.py](CMK_OCSVM_scatter.py)

- **散度损失** `scatter_loss`（[CMK_OCSVM_scatter.py:67](CMK_OCSVM_scatter.py#L67)）,借鉴 CMKKM(质心多核 K-means)的单类目标:

$$
L_{scat} = -\frac{1}{K}\sum_{k=1}^{K}\lVert \mu_k \rVert^2,
\qquad \mu_k = \frac{1}{N}\sum_{i=1}^{N} h_{k,i}
$$

  在 L2 归一化嵌入 + 线性核下严格等价于:

$$
-\frac{1}{K}\sum_k \frac{1}{N^2}\sum_{i,j} h_{k,i}\cdot h_{k,j}
= -\frac{1}{K}\sum_k (\text{核 } k \text{ 内正常样本对的平均余弦相似度})
$$

  即**最大化嵌入重心模长 ≡ 最大化正常样本对的平均余弦相似度 ≡ 最小化正常样本在各核潜空间中的散度**。

- **总损失**（[CMK_OCSVM_scatter.py:130](CMK_OCSVM_scatter.py#L130)）:

$$
L_{total} = L_{cross} + \lambda \cdot L_{scat}
$$

  `λ = 0` 时精确退化为原始 CMK,便于消融。

- **复杂度优势**:用嵌入重心模长替代显式逐对相似度,计算量为 **O(K·N·d)**,远优于显式 O(K·N²·d)。

- **防坍缩保证**（[CMK_OCSVM_scatter.py:26-29](CMK_OCSVM_scatter.py#L26)）:
  - 超球面给出上界 `‖μ_k‖ ≤ 1`;
  - 嵌入全部坍缩为同一点时,InfoNCE 梯度仍非零(loss = log 2N);
  - 两项损失形成张力,天然排除退化解。

---

## 贡献 3 ─ 方向 + 幅值双信号检测器(max_ensemble)

文件:[hybrid_score/run_hybrid_score.py](hybrid_score/run_hybrid_score.py)

代码注释（[run_hybrid_score.py:5-14](hybrid_score/run_hybrid_score.py#L5)）点明了一个被既有方法忽视的观察 —— 判别结构分为两类:

| 信号 | 代表数据 | 现象 | 对应通道 |
|---|---|---|---|
| **方向**(余弦) | ionosphere | 正常/异常方向差异大,幅值无差异 | L2 归一化嵌入 (N, K·d) → **linear** OC-SVM |
| **幅值**(范数) | wbc | 训练后投影幅值差 7.38×,余弦域几乎无差异 | 各核原始范数 (N, K) → **RBF** OC-SVM |

- `extract_components`（[run_hybrid_score.py:74](hybrid_score/run_hybrid_score.py#L74)）一次前向同时取出**归一化嵌入(方向信号)**与**各核范数(幅值信号)**。
- `ensemble_scores`（[run_hybrid_score.py:120](hybrid_score/run_hybrid_score.py#L120)）:两路各自做 nu 网格寻优 → min-max 归一化到 [0,1] → **逐样本取 max** 融合。

**关键价值:** 单纯 L2 归一化(贡献 1/2 的默认做法)会**丢弃幅值信号**,在 wbc 类数据上失效;max_ensemble 在每个样本上自适应挑出更强的那一路信号,把方向法与幅值法的优势并集起来。

---

## 一句话概括

> **SCMK = 多尺度跨核对比一致性(CMK) + 多核散度紧凑性惩罚(scatter,核心) + 方向/幅值双信号自适应融合检测(max_ensemble)。**

三者分别解决"多视角表示""正常簇紧致""单一信号失配"三个问题,且 scatter 项具备 CMKKM 等价推导与防坍缩理论支撑。

---

## 训练/评估配置(代码实证)

| 项 | 取值 | 出处 |
|---|---|---|
| 核数 K | 5(高斯,ratios 0.1/0.5/1/2/5) | [CMK_OCSVM.py:136](CMK_OCSVM.py#L136) |
| 嵌入维度 d | {16, 32, 64, 128, 256} | [CMK_OCSVM.py:51](CMK_OCSVM.py#L51) |
| 散度权重 λ | {0, 0.1, 1, 10, 100, 1000} | [hybrid_score/run_hybrid_score.py:66](hybrid_score/run_hybrid_score.py#L66) |
| OC-SVM ν | {0.01, 0.05, 0.1, 0.2} | [CMK_OCSVM.py:55](CMK_OCSVM.py#L55) |
| 优化器 | Adam, lr=0.01, batch=512, epochs=100, seed=42 | [CMK_OCSVM.py:65](CMK_OCSVM.py#L65) |
| 训练数据 | 仅正常样本(`y==0`),纯半监督 | [CMK_OCSVM_scatter.py:112](CMK_OCSVM_scatter.py#L112) |
