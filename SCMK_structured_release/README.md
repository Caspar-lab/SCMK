# SCMK reproducibility package

This directory contains the SCMK code, three compact example datasets,
pretrained projection heads, and per-sample anomaly scores used to reproduce
the results reported in the manuscript. Glass, Ecoli, and WBC are included for
end-to-end execution. The archived outputs for all 20 datasets remain available
for exact numerical verification.

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

The `scores.csv` files stored beside the projection weights are retained as
provenance records. Developers can audit their completeness with
`python verify_checkpoint_outputs.py`; this optional check is not checkpoint
inference.

## 2. End-to-end inference from pretrained checkpoints

Glass, Ecoli, and WBC include compact, complete inference checkpoints that do
not contain final per-sample anomaly scores. For example:

```powershell
python evaluate_full_checkpoint.py --dataset glass --seed 0
python evaluate_full_checkpoint.py --dataset ecoli --seed 1
python evaluate_full_checkpoint.py --dataset wbc --seed 2
```

These commands load the projection heads, fitted directional and magnitude
OC-SVMs, kernel bandwidths, directional training embeddings, and split indices.
They then regenerate both anomaly signals and their fused AUC without fitting
either the representation model or the detectors. All nine example runs (three
datasets by three seeds) reproduce their expected AUC exactly.

The complete example files and further instructions are under
`examples/pretrained_pipeline/`.

## 3. Run SCMK from scratch

Run one bundled dataset and split:

```powershell
python main.py --dataset glass --seed 0
```

Run all three bundled examples:

```powershell
python run_experiments.py --datasets glass ecoli wbc_malignant_39_variant1 --seeds 0 1 2
```

New scores are written to `outputs/<dataset>/seed<seed>/`. Existing results are
skipped unless `--force` is supplied. Use `--device cpu`, `--device cuda`, or
`--device cuda:0` to override automatic device selection.

The configuration file and archived scores cover all 20 manuscript datasets,
but the remaining raw datasets are not redistributed in this branch. To retrain
the complete benchmark, obtain those datasets from their original sources and
place them in the paths described by `data/raw/` and `data/splits/`.

## 4. Regenerate the released checkpoints

Regenerate the checkpoints for the three bundled datasets with:

```powershell
python export_checkpoints.py --datasets glass ecoli wbc_malignant_39_variant1
```

The export is resumable. To regenerate selected runs:

```powershell
python export_checkpoints.py --datasets glass --seeds 0 1 2 --force
```

Retraining can vary slightly with PyTorch, CUDA, and numerical-library
versions. Therefore, `reference_scores/` remains authoritative for the exact
manuscript values, whereas `checkpoints/` records the released trained weights
and their paired outputs.

Running `python export_checkpoints.py` without `--datasets` requests all 20
datasets and therefore requires the additional raw datasets and fixed split
files to be supplied first.

## Repository layout

```text
SCMK_structured_release/
|-- data/
|   `-- raw/                  Glass, Ecoli, and WBC example datasets
|-- model/                    SCMK and bounded-memory implementations
|-- utils/                    configuration, data, evaluation, and run helpers
|-- checkpoints/              60 pretrained runs and paired scores
|-- examples/                 complete no-training inference examples
|-- reference_scores/         exact manuscript per-sample scores
|-- configs.json              final dataset- and seed-specific settings
|-- default.yaml              shared training and detector settings
|-- main.py                   single-run entry point
|-- run_experiments.py        multi-run training entry point
|-- export_checkpoints.py     checkpoint regeneration
|-- verify_checkpoint_outputs.py optional archived-output integrity check
|-- evaluate_full_checkpoint.py end-to-end checkpoint inference
|-- export_full_examples.py   regenerate compact inference examples
|-- verify_auc.py             exact manuscript AUC verification
`-- requirements.txt
```

## Experimental protocol

- Seeds 0, 1, and 2 determine the normal train/test split. Network
  initialization remains fixed at seed 42.
- For the bundled datasets, half of the normal samples are used for training;
  the remaining normal samples and all anomalies form the test set.
- In the full manuscript benchmark, Nursery, Shuttle, and SMTP use fixed split
  files. Those large-dataset files are not redistributed in this branch.
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
