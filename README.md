# exp-driven-mario-level

Training pipelines are organized by architecture:

- [Initial architecture](model-training/initial_model_architecture.md): four stages, GroupNorm, LeakyReLU.
- [Model architecture 1](model-training/model_arch_1.md): three stages, LayerNorm, ReLU.
- [Model architecture 2](model-training/model_arch_2.md): four stages, LayerNorm, ReLU.

Each also has a variant with softmax tile predictions and a separate sigmoid player-path prediction: [initial_model_architecture_softmax](model-training/initial_model_architecture_softmax/README.md), [model_arch_1_softmax](model-training/model_arch_1_softmax/README.md), and [model_arch_2_softmax](model-training/model_arch_2_softmax/README.md). These use tile categorical cross-entropy plus path binary cross-entropy for reconstruction. The original three pipelines retain binary cross-entropy over all channels.

Six additional pipelines multiply a cell's tile reconstruction loss when its actual tile is another type but the decoded prediction is ground or empty air. Their fifth hyperparameter, `background_error_weight`, controls that multiplier; the supplied rows use `3`, and `1` disables the extra penalty. The independent player-path loss keeps its usual weight.

| Architecture | Weighted original BCE | Weighted softmax |
|---|---|---|
| Initial | [initial_model_architecture_weighted](model-training/initial_model_architecture_weighted/README.md) | [initial_model_architecture_softmax_weighted](model-training/initial_model_architecture_softmax_weighted/README.md) |
| Architecture 1 | [model_arch_1_weighted](model-training/model_arch_1_weighted/README.md) | [model_arch_1_softmax_weighted](model-training/model_arch_1_softmax_weighted/README.md) |
| Architecture 2 | [model_arch_2_weighted](model-training/model_arch_2_weighted/README.md) | [model_arch_2_softmax_weighted](model-training/model_arch_2_softmax_weighted/README.md) |

Each folder has its own model, training entry point, editable hyperparameter file, and default run outputs. Data loading, plotting, reconstruction previews, and saved-run analysis use shared modules in `model-training/`. Choose a folder and run its `train.py` using the shared root environment. For example, after the setup in [TRAINING.md](model-training/TRAINING.md):

```bash
cd model-training/model_arch_1
../../.venv/bin/python train.py
../../.venv/bin/python ../analyze_runs.py --runs-dir runs
```

See the [training guide](model-training/TRAINING.md) for all commands and the [saved-run analysis guide](model-training/ANALYZE_RUNS.md) for ranking completed epochs.
