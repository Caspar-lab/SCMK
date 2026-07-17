# SCMK loss-architecture experiments

Everything added on top of the original SCMK method (vs git `90bf2ef`): a debug
replica of the semi-supervised pipeline that adds a **temperature τ** and
**embedding-calibrated bandwidths** to the cross-kernel contrastive loss, plus the
batch experiments built on it. All new work lives here; outputs go to
`result/scmk_experiments/`.

## Scripts (this folder)

| script | what it does | output (`result/scmk_experiments/`) |
|--------|--------------|-------------------------------------|
| `run_hybrid_score_semi_debug.py` | shared module: SCMK loss + τ (`logits=exp(K_avg/τ)`) + `bw_mode∈{raw,embedding}`; single-dataset 2×2 factorial when run directly | (trace only) |
| `temp_sweep_emb.py` | **τ sweep with EMBEDDING bandwidth**, best-τ per dataset, vs original SCMK | `temp_sweep_emb_results.csv` / `.log` |
| `temp_sweep_all.py` | τ sweep with RAW bandwidth (isolates the temperature axis; τ=1 ≡ SCMK) | `temp_sweep_results.csv` / `temp_sweep.log` |
| `factorial_all.py` | 2×2 factorial `{raw,emb} bw × {1, 0.2} τ` across datasets | `factorial_all_results.csv` / `.log` |
| `debug_cosine_vs_kernel.py` | cosine NT-Xent vs the Gaussian kernel-logit loss | `cosine_vs_kernel_results.csv` / `.log` |
| `build_experiments_excel.py` | compiles everything below into one workbook | `SCMK_experiments_summary.xlsx` |
| `factorial_legend.md` | methodology legend: the two axes (loss version vs detector) | — |

All batch scripts fix `dim=64, λ=100, seed=2` on the 19 manuscript datasets
(the 20 minus pageblocks). Run with `conda run -n torch311 python <script>`.

## The compiled workbook — `SCMK_experiments_summary.xlsx`

| sheet | contents |
|-------|----------|
| `Overview` | the two axes, per-sheet index, caveats |
| `new_vs_SCMK_summary` | per-dataset head-to-head: original SCMK loss vs each new loss variant (fused AUC) |
| `new_temp_sweep_emb` / `_raw` | τ sweeps (fused + emb-MK), with best-τ columns |
| `new_factorial_2x2` | bandwidth × temperature factorial |
| `new_cosine_vs_kernel` | cosine vs kernel-logit loss |
| `SCMK_param_effect` | **original** SCMK: marginal effect of dim / λ / seed + dim×λ pivot |
| `SCMK_best` / `SCMK_grid_mean` / `SCMK_grid_all` | original SCMK study: best config / seed-mean / all 9000 runs (dim×λ×seed) |

**Headline (mean fused AUC over 19 datasets, oracle best-τ):** original SCMK
0.831 → emb-bw best-τ 0.877 (+0.046), raw-bw best-τ 0.879 (+0.048), factorial best
cell 0.866 (+0.035), cosine best 0.821 (−0.010). Best-τ is an **oracle** (chosen on
test AUC) = tuning upper bound; the deployable number is the best single global τ.
