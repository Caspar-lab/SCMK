# stat — 统计显著性检验（Friedman / Nemenyi / Wilcoxon）

对 `selection.csv`（20 数据集 × {9 对比算法 + SCMK} 的 AUC）做统计显著性检验，
方法对齐 `C:\OD\Shihao\stat` 的 MATLAB `criticaldifference2` 流程（Demšar 2006）。

## 文件

| 文件 | 作用 |
|------|------|
| `stat_test.py` | 检验主脚本：Friedman 检验 + Nemenyi 后续检验 + Critical Difference (CD) 图（matplotlib，MATLAB 配色：轴黑/算法蓝/CD 红）+ 成对 Wilcoxon signed-rank（Holm 校正）。读 `../selection.csv` |
| `SCMK_stat.xlsx` | 与 `GMKAD_results.xlsx` 同布局（第 1 行算法名，20 行数据集 AUC，SCMK 在末列），供 MATLAB `criticaldifference2` 直接出版级出图 |
| `cd_diagram.pdf` / `.png` | CD 图（Nemenyi, α=0.05）。也是论文 manuscript.tex 的 Figure 1 来源 |
| `stat_report.txt` | 完整文字结论（Friedman/平均秩/CD/逐算法 Wilcoxon）|

## 主要结论

- **Friedman**：χ²=70.42, p=1.3×10⁻¹¹ → 十个方法间存在极显著差异
- **平均秩**：SCMK = 1.00（20 数据集全部第一），最接近的 DIF = 4.35
- **Nemenyi**：CD(0.05)=3.03；SCMK 与所有 9 个算法的秩差 > CD → 显著优于全部
- **Wilcoxon + Holm**：9 个对比全部 p_holm < 0.001（`***`），SCMK 20/20 全胜

## 运行

```
python stat_test.py        # 用含 openpyxl 的环境（base anaconda）
```

> MATLAB 出版图：用 `SCMK_stat.xlsx` 跑 `criticaldifference2`，注意把原脚本的
> `Range 'A1:N1'`（14 列）改为 `'A1:J1'`（本数据 10 列）。
