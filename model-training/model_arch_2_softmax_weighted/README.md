# model_arch_2_softmax_weighted

This pipeline uses the network in [the base architecture](../model_arch_2_softmax.md) and tile cross-entropy plus path BCE. It multiplies tile reconstruction errors by `background_error_weight` only when a cell whose actual tile is outside ground/air is decoded as ground (`X` or `#`) or air (`-`). The path channel keeps its usual loss weight.

The gate uses the highest of the 21 tile logits, with a hard argmax treated as constant during backpropagation. Losses retain their fixed cell/bit denominators. Weight `1` reproduces the base objective; larger weights penalize these mistakes more heavily. Network layers and parameter counts match the base model; the checkpoint architecture version appends `_weighted`.

Edit `hyperparameter-combinations.txt`. Each row has five values:

```text
learning_rate, lambda_rec, lambda_kl, lambda_rank, background_error_weight
```

Rows inherit this base pipeline's current combinations, with `3` appended. The last value must be finite and at least `1`. `lambda_rec` still weights the complete reconstruction term once.

After the shared setup in [TRAINING.md](../TRAINING.md), run from the repository root:

```bash
cd model-training/model_arch_2_softmax_weighted
../../.venv/bin/python train.py
../../.venv/bin/python ../analyze_runs.py --runs-dir runs
```

All existing training flags, saved fitting/validation splits, checkpoint ranges, and reconstruction previews remain available. Outputs default to this folder's `runs/` and `run-analysis/`. The fifth value is included in combination directory names, saved configurations, checkpoints, reports, and plots.
