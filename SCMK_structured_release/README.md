# SCMK structured reproducibility package

This directory is a structured copy of the SCMK code and data corresponding to
the AUC results in the current manuscript. It does not depend on the original
experimental workspace.

## Layout

```text
SCMK_structured_release/
├── data/
│   ├── raw/                 # 17 datasets split deterministically at runtime
│   └── splits/              # fixed seed-0/1/2 splits for Nursery, Shuttle, SMTP
├── model/
│   ├── scmk.py              # core SCMK implementation
│   ├── legacy.py            # exact ordinary-dataset manuscript pipeline
│   └── bounded.py           # bounded-memory Shuttle/SMTP pipeline
├── utils/
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   └── runner.py
├── reference_scores/        # 20 datasets x 3 seeds, per-sample scores
├── configs.json             # dataset- and seed-specific manuscript settings
├── default.yaml             # shared defaults (JSON-compatible YAML)
├── main.py                  # one dataset/seed
├── run_experiments.py       # multiple or all datasets/seeds
├── verify_auc.py            # recompute AUC without retraining
└── requirements.txt
```

## Installation

```powershell
python -m pip install -r requirements.txt
```

Install a CUDA-specific PyTorch wheel separately when GPU acceleration is
required. The reference environment used Python 3.11 and PyTorch 2.11.0 with
CUDA 12.8.

## Verify the manuscript AUC values

This is fast and does not retrain the model:

```powershell
python verify_auc.py
```

The command reads the cached per-sample scores and checks every dataset mean and
sample standard deviation against the three-decimal manuscript values.

## Run one experiment

```powershell
python main.py --dataset thyroid --seed 0
```

The output is written to `outputs/<dataset>/seed<seed>/SCMK.csv`, accompanied by
a JSON file containing the configuration, environment, AUC, reference AUC, and
difference.

## Run several or all experiments

```powershell
python run_experiments.py --datasets glass lymphography --seeds 0 1 2
python run_experiments.py
```

Existing outputs are skipped unless `--force` is supplied. Use `--device cpu`,
`--device cuda`, or `--device cuda:0` to override automatic device selection.

## Experimental protocol

- Seeds 0, 1, and 2 determine the normal train/test split; network initialization
  remains fixed at seed 42, matching the recorded experiments.
- For the 17 ordinary datasets, half of the normal samples are selected for
  training and the remaining normals plus all anomalies form the test set.
- Nursery, Shuttle, and SMTP use the stored fixed split files.
- Shuttle and SMTP use the bounded-memory pipeline: at most 8,000 normal training
  samples, training batch size 256, bandwidth sample size 4,096, and score block
  size 1,024.
- The two OC-SVM branches select `nu` independently from
  `{0.01, 0.05, 0.1, 0.2}` using the recorded manuscript protocol.
- `configs.json` is authoritative for dataset- and seed-specific representation
  parameters. In particular, Shuttle seed 2 uses a different scatter weight
  from seeds 0 and 1, as required by the current cached result.

Retraining can be sensitive to numerical library, CUDA, and hardware versions.
`reference_scores/` and `verify_auc.py` provide the exact auditable connection
to the manuscript AUC table, while newly trained scores are always written to a
separate output directory.

## Export and use pretrained checkpoints

The final dataset- and seed-specific configurations can be retrained once while
exporting the learned projection heads:

```powershell
python export_checkpoints.py
```

The command is resumable. Each `checkpoints/<dataset>/seed<seed>/` directory
contains `projection_heads.pt`, `scores.csv`, and `scores.json`. Full kernel
Gram matrices are deliberately omitted because their quadratic space cost is
unsuitable for GitHub, especially for Shuttle and SMTP.

After the export finishes, obtain all checkpoint-run AUC values without neural
network training:

```powershell
python evaluate_pretrained.py
```

Three reproduction levels are therefore available:

1. `verify_auc.py` checks the exact archived manuscript scores in seconds.
2. `evaluate_pretrained.py` checks results paired with the released weights.
3. `run_experiments.py` performs complete training from scratch.

The archived scores remain the authoritative source for the exact manuscript
numbers because retraining can vary slightly across PyTorch, CUDA, and
numerical-library versions.
