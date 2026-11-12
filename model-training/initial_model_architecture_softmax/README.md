# Initial architecture: softmax training pipeline

This variant uses softmax over the 21 tile channels and an independent sigmoid player-path channel for reconstruction. Training combines mean tile categorical cross-entropy and mean path binary cross-entropy. See the concise [architecture differences](../initial_model_architecture_softmax.md) and the inherited [initial architecture](../initial_model_architecture.md).

After the shared environment setup in [TRAINING.md](../TRAINING.md), run from the repository root:

```bash
cd model-training/initial_model_architecture_softmax
../../.venv/bin/python train.py
../../.venv/bin/python ../analyze_runs.py --runs-dir runs
```

Edit this folder's `hyperparameter-combinations.txt` to choose runs, or supply `--hyperparameters-file`. CLI flags, checkpoint windows, and validation previews follow the shared training guide. Outputs default to this folder's `runs/`; analysis defaults to `run-analysis/`. The original `initial_model_architecture` pipeline keeps its existing all-channel BCE objective.
