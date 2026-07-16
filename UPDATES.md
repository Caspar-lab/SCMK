# Updates since the last commit

Baseline commit: **`90bf2ef` — "修改摘要和引言" (2026-07-15 22:19)**.
This file records every change in the working tree relative to that commit.
Generated from `git status` + the actual diffs; grouped by topic, with a raw
file inventory at the end.

---

## 1. Manuscript (`CMK_OCSVM_scatter_latex/elsarticle/manuscript.tex`)

Two additions to the **Experiments → Ablation study** section (`\S4.6`,
`subsec:exp-ablation`). Net: +106 / −31 lines. Recompiled cleanly (35 pages).

### 1a. Table 5 expanded from 8 → 20 datasets (four-core-component ablation)
- Rewrote `tab:ablation` to cover **all twenty benchmark datasets** (previously
  only eight), matching the dataset set of Tables 2/3/6.
- Protocol unchanged: seed-2 split, oracle-best over the dim×λ grid, six variants
  (Full, w/o $\mathcal{L}_{\text{scat}}$, w/o $\mathcal{L}_{\text{con}}$,
  single-kernel, dir.-only, mag.-only). Overlapping datasets reproduce the old
  values exactly (Cardio 0.967, WBC 0.985, Ionosphere 1.000, …).
- New average AUCs: **Full 0.932**, w/o scatter 0.793 (−0.139), w/o cross 0.893
  (−0.039), single-kernel 0.872 (−0.060), dir.-only 0.917 (−0.015), mag.-only
  0.892 (−0.041).
- Updated the caption and the three finding paragraphs (scatter dominance, the
  dual-signal robustness argument, and the multi-kernel/contrastive terms) with the
  new numbers and dataset examples (Vowels/Sonar, no longer in the set, replaced by
  Bands-6/Glass/Audiology/Thyroid, etc.).

### 1b. Table 6 added — learnable projection-head ablation (`tab:projhead`)
- New paragraph "The learnable projection head is essential." + a 20-dataset,
  3-seed mean±std table comparing **Full** vs **Raw-MK** (multi-kernel similarity
  on raw features) vs **Direct-OCSVM** (plain RBF OC-SVM).
- Result: Full **0.914**, Raw-MK 0.647 (−0.267), Direct-OCSVM 0.843 (−0.071);
  Full best on 14/20. Validates that the learned projection—not the kernel bank or
  the OC-SVM head alone—drives performance.

---

## 2. Component ablation experiment (`Granular-CMK/hybrid_score/ablation/`)

- **`run_ablation.py`** — dataset list changed from the 8 representative sets to
  the full 20 manuscript datasets (Table-2 order). Protocol otherwise unchanged.
- **`ablation_long.csv`, `ablation_wide.csv`** — regenerated for the 20 datasets
  (source of Table 5). Run log: **`run20.log`**.
- **`ablation_long.8ds.bak.csv`, `ablation_wide.8ds.bak.csv`** — backups of the
  previous 8-dataset results (kept for provenance).

---

## 3. Projection-head ablation (`projection_head_ablation/`, new folder)

Validates SCMK's learnable multi-kernel projection head. Core study + several
follow-up diagnostics:

- **`run_projhead_ablation.py`** + `results/projhead_ablation.csv`,
  `results/projhead_table.tex` — the main study (source of Table 6): 20 datasets,
  seeds {0,1,2}. `Full` is taken from the AUC experiments
  (`master_compare_v2.csv`); `Raw-MK` and `Direct-OCSVM` are computed fresh under
  the identical protocol.
- **`raw_mk.py`** — standalone Raw-MK variant (removes the projection head, feeds
  the raw-feature multi-kernel empirical map into the same dual-signal OC-SVM head)
  for inspecting the workflow.
- **`emb_mk.py`** + `results/embmk_variant.csv` — additional variant applying
  `gauss_med_kernels` to the **learned** embeddings (Emb-MK), reported alongside
  Full and Raw-MK to confirm the multi-kernel scheme is meaningful on the learned
  space.
- **`run_extended_nu.py`** + `results/extended_nu.csv`, `extended_nu_table.tex` —
  extended OC-SVM `nu` sweep, added because the original grid
  `[0.01,0.05,0.1,0.2]` made the precomputed learned-kernel variant tie the plain
  raw-RBF OC-SVM; the sweep shows that tie is partly a grid artifact.
- **`README.md`** — write-up of the main projection-head study.

---

## 4. Other new experiments

- **`Granular-CMK/hybrid_score/ab_combined_kernel_semi.py`** +
  `result/hybrid_score_semi/ab_combined_kernel_semi.csv` — A/B comparison of the
  reference paper's "learned kernel matrix" downstream (from
  `CMK-code_release/CMK.py`'s `ConLoss`) against the current `max_ensemble` scoring,
  under the semi-supervised split.
- **`DMFAD/`** (new directory) — DMFAD implementation and result-extraction scripts
  (`DMFAD.py`, `Extract singleAlg_DMFAD_ablation.py`,
  `Extract the results_singleAlg.py`).

---

## 5. Reference material & build artifacts

- **`Contrastive_Multi-View_Kernel_Learning.pdf`** — reference paper added to the
  repo.
- **`CMK-code_release`** — reference implementation directory shows modified content.
- LaTeX build outputs regenerated: `manuscript.pdf` (716 KB → 830 KB),
  `manuscript.aux`, `manuscript.log`, `compile.log`, and new
  `manuscript.fdb_latexmk` / `manuscript.fls`. Stale artifacts removed
  (`manuscript.synctex.gz`, `method.aux`, `method.synctex.gz`).
- Recompiled `*.pyc` caches under `Granular-CMK/**/__pycache__/`.

---

## Raw file inventory (`git status`)

```
Modified (tracked):
  CMK_OCSVM_scatter_latex/elsarticle/manuscript.tex        (+106 / -31)
  CMK_OCSVM_scatter_latex/elsarticle/manuscript.pdf/.aux/.log
  CMK_OCSVM_scatter_latex/elsarticle/compile.log
  Granular-CMK/hybrid_score/ablation/run_ablation.py       (8 -> 20 datasets)
  Granular-CMK/hybrid_score/ablation/ablation_long.csv
  Granular-CMK/hybrid_score/ablation/ablation_wide.csv
  CMK-code_release                                          (submodule/dir content)
  Granular-CMK/**/__pycache__/*.pyc

Deleted (build artifacts):
  CMK_OCSVM_scatter_latex/elsarticle/manuscript.synctex.gz
  CMK_OCSVM_scatter_latex/elsarticle/method.aux
  CMK_OCSVM_scatter_latex/elsarticle/method.synctex.gz

Untracked (new):
  projection_head_ablation/                    (run_projhead_ablation.py, raw_mk.py,
                                                emb_mk.py, run_extended_nu.py,
                                                README.md, results/*.csv, *.tex)
  Granular-CMK/hybrid_score/ab_combined_kernel_semi.py
  result/hybrid_score_semi/ab_combined_kernel_semi.csv
  Granular-CMK/hybrid_score/ablation/ablation_long.8ds.bak.csv
  Granular-CMK/hybrid_score/ablation/ablation_wide.8ds.bak.csv
  Granular-CMK/hybrid_score/ablation/run20.log
  DMFAD/
  Contrastive_Multi-View_Kernel_Learning.pdf
  CMK_OCSVM_scatter_latex/elsarticle/manuscript.fdb_latexmk, manuscript.fls
```

> Note: three Introduction citations (`chen2024coarse`, `xie2021network`,
> `du2025fraud`) remain undefined (no matching `\bibitem`) — pending bibliography
> entries.
