# Initial architecture training pipeline

Four encoder/decoder stages with GroupNorm, LeakyReLU(0.2), and FC-Tail dropout 0.1 after the first hidden layer. Checkpoints record `architecture_version="initial_model_architecture_crop16"`. See the [architecture specification](../initial_model_architecture.md).

Run from this folder after installing [the common dependencies](../requirements.txt) in the repository's root `.venv`:

```bash
../../.venv/bin/python train.py
```

Edit this folder's [hyperparameter-combinations.txt](hyperparameter-combinations.txt) to choose its runs. Each listed row trains a fresh model for 50 epochs by default. This folder's `train.py` always uses this folder's [model.py](model.py). Its training constants are independent of the other model folders.

Default outputs go to `runs/<UTC-timestamp>/` here. Checkpoint saving is disabled by default; to save epochs 30 through 40 inclusive:

```bash
../../.venv/bin/python train.py \
  --checkpoint-start-epoch 30 \
  --checkpoint-end-epoch 40
```

Analyze this folder's saved runs with the shared analysis script:

```bash
../../.venv/bin/python ../analyze_runs.py --runs-dir runs
```

Default analysis outputs go to `run-analysis/` here. The dataset and rendering assets are shared with the repository, and their default paths work from any working directory. Explicit relative CLI paths resolve from your current directory.

From the repository root, the equivalent training command is:

```bash
.venv/bin/python model-training/initial_model_architecture/train.py
```

This folder keeps the architecture-specific [model](model.py), [training entry point](train.py), and hyperparameters. All architectures reuse one copy of [data loading](../data.py), [curve plotting](../plotting.py), [checkpoint previews](../reconstruction_previews.py), and [saved-run analysis](../analyze_runs.py) from `model-training/`.

See [TRAINING.md](../TRAINING.md) for setup, a short smoke run, dataset conventions, losses, checkpoint loading, and all CLI settings. See [ANALYZE_RUNS.md](../ANALYZE_RUNS.md) for ranking options and analysis of earlier root-level runs.
