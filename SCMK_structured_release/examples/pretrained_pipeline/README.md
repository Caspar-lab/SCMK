# End-to-end pretrained inference examples

Glass, Ecoli, and WBC are provided as compact examples of inference from a
complete pretrained SCMK pipeline. Unlike `reference_scores/`, these examples
do not store final per-sample anomaly scores.

Each seed directory contains the learned projection heads, two fitted OC-SVM
detectors, directional training embeddings and bandwidths, split indices, and
metadata. Run an example from the package root:

```powershell
python evaluate_full_checkpoint.py --dataset glass --seed 0
python evaluate_full_checkpoint.py --dataset ecoli --seed 1
python evaluate_full_checkpoint.py --dataset wbc --seed 2
```

The script loads the original data, applies the pretrained projections,
reconstructs the cross-kernel similarities, evaluates both fitted detectors,
fuses their scores, and computes AUC. It performs no representation or OC-SVM
training.
