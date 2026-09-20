# SCMK reproducibility package

This directory contains the code, datasets, pretrained projection heads, and
per-sample anomaly scores used to reproduce the SCMK results reported in the
manuscript. It is self-contained and does not depend on other repository files.

## Quick start

Run the following commands from this directory:

```powershell
cd SCMK_structured_release
conda create -n scmk python=3.11 -y
conda activate scmk
python -m pip install -r requirements.txt
```

Install a platform-appropriate PyTorch build separately if the wheel specified
in `requirements.txt` is unavailable for the local CUDA version.

## 1. Reproduce the manuscript AUC table without training

The fastest and exact reproduction path uses the released per-sample scores:

```powershell
python verify_auc.py
```

The script recomputes the AUC of every dataset and split from
`reference_scores/` and checks the three-split statistics against the
manuscript. Expected final output:

```text
Mean AUC across datasets: 0.948848742176
All manuscript rows match: True
```

## 2. Verify the released checkpoint runs without training

Each `checkpoints/<dataset>/seed<seed>/` directory contains:

```text
projection_heads.pt   learned PyTorch projection-head weights
scores.csv            sample indices, labels, and anomaly scores
scores.json           configuration, environment, and AUC metadata
```

Verify all checkpoint-associated results with:

```powershell
python evaluate_pretrained.py
```

The expected overall mean is `0.948848742176`. The paired score files make the
reported AUC directly auditable without retraining or storing prohibitively
large full Gram matrices.

## 3. Run SCMK from scratch

Run one dataset and split:

```powershell
python main.py --dataset thyroid --seed 0
```

Run selected datasets or the complete benchmark:

```powershell
python run_experiments.py --datasets glass lymphography --seeds 0 1 2
python run_experiments.py
```

New scores are written to `outputs/<dataset>/seed<seed>/`. Existing results are
skipped unless `--force` is supplied. Use `--device cpu`, `--device cuda`, or
`--device cuda:0` to override automatic device selection.

## 4. Regenerate the released checkpoints

Retrain every final dataset-specific configuration and export new projection
weights and paired scores with:

```powershell
python export_checkpoints.py
```

The export is resumable. To regenerate selected runs:

```powershell
python export_checkpoints.py --datasets glass --seeds 0 1 2 --force
```

Retraining can vary slightly with PyTorch, CUDA, and numerical-library
versions. Therefore, `reference_scores/` remains authoritative for the exact
manuscript values, whereas `checkpoints/` records the released trained weights
and their paired outputs.

## Repository layout

```text
SCMK_structured_release/
|-- data/
|   |-- raw/                  17 datasets split at runtime
|   `-- splits/               fixed splits for Nursery, Shuttle, and SMTP
|-- model/                    SCMK and bounded-memory implementations
|-- utils/                    configuration, data, evaluation, and run helpers
|-- checkpoints/              60 pretrained runs and paired scores
|-- reference_scores/         exact manuscript per-sample scores
|-- configs.json              final dataset- and seed-specific settings
|-- default.yaml              shared training and detector settings
|-- main.py                   single-run entry point
|-- run_experiments.py        multi-run training entry point
|-- export_checkpoints.py     checkpoint regeneration
|-- evaluate_pretrained.py    checkpoint-result AUC verification
|-- verify_auc.py             exact manuscript AUC verification
`-- requirements.txt
```

## Experimental protocol

- Seeds 0, 1, and 2 determine the normal train/test split. Network
  initialization remains fixed at seed 42.
- For the 17 ordinary datasets, half of the normal samples are used for
  training; the remaining normal samples and all anomalies form the test set.
- Nursery, Shuttle, and SMTP use the included fixed split files.
- Shuttle and SMTP use the bounded-memory implementation with at most 8,000
  normal training samples, training batch size 256, bandwidth sample size
  4,096, and scoring block size 1,024.
- The five Gaussian-kernel bandwidth ratios are `{0.1, 0.5, 1, 2, 5}`.
- The directional and magnitude OC-SVM branches independently select `nu` from
  `{0.01, 0.05, 0.1, 0.2}` under the recorded manuscript protocol.
- `configs.json` is authoritative for the dataset- and seed-specific
  representation dimension, scatter weight, and contrastive temperature.

## Loading a projection checkpoint

```python
import torch

checkpoint = torch.load(
    "checkpoints/annealing_variant1/seed0/projection_heads.pt",
    map_location="cpu",
    weights_only=False,
)
state_dict = checkpoint["state_dict"]
print(checkpoint["input_dim"], checkpoint["latent_dim"])
print(state_dict.keys())
```

The checkpoint contains projection-head weights rather than optimizer state. It
is intended for representation extraction and result inspection, not for
resuming an interrupted optimizer trajectory.
