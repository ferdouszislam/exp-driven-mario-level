# exp-driven-mario-level

Experience-driven Mario level generation using convolutional variational autoencoders (CNN-VAEs), player paths, and pairwise human fun preferences. This project includes dataset preparation, model training, experiment analysis, and level reconstruction and generation.

## Project structure

```text
exp-driven-mario-level/
├── datasets/
│   └── fun-study-dataset-analysis/          # Dataset documentation and tools
│       ├── Mario-Fun-Study/                # Source levels, player traces, and annotations
│       ├── preprocessed_dataset/           # Prepared training, validation, and test pairs
│       ├── encoding_level_images/          # Rendered dataset examples
│       ├── preprocess_dataset.py          # Build the training representation and splits
│       ├── encoding_to_level_image.py      # Render encoded levels and paths
│       ├── baseline_metrics.py             # Calculate reconstruction and ranking baselines
│       └── tests/                          # Preprocessing and rendering checks
├── model-training/
│   ├── initial_model_architecture/         # Base architecture
│   ├── initial_model_architecture_softmax/
│   ├── initial_model_architecture_weighted/
│   ├── initial_model_architecture_softmax_weighted/
│   ├── model_arch_1/                       # Alternative architecture 1
│   ├── model_arch_1_softmax/
│   ├── model_arch_1_weighted/
│   ├── model_arch_1_softmax_weighted/
│   ├── model_arch_2/                       # Alternative architecture 2
│   ├── model_arch_2_softmax/
│   ├── model_arch_2_weighted/
│   ├── model_arch_2_softmax_weighted/
│   ├── data.py                            # Shared dataset loading
│   ├── plotting.py                        # Training curves
│   ├── analyze_runs.py                    # Compare saved experiment results
│   ├── generate_levels.py                 # Reconstruct and generate from checkpoints
│   ├── reconstruction_previews.py          # Shared preview helpers
│   └── tests/                             # Model and pipeline checks
├── logs/                                  # Slurm job logs
├── run_models.sh                          # Cluster-specific Slurm launcher for all 12 models
├── requirements.txt                       # Shared Python dependencies
└── LICENSE
```

Each model folder contains its own `model.py`, `train.py`, and `hyperparameter-combinations.txt`. The `_softmax` variants use categorical tile predictions; `_weighted` variants add tile reconstruction penalties; `_softmax_weighted` combines both. Architecture specifications and shared utilities live directly in `model-training/`.

Training outputs normally appear in each model folder's `runs/`, with analysis reports in `run-analysis/` beside it. Existing `runs*` and `run-analysis*` folders directly under `model-training/` also contain saved experiments and reports.

## Where to start

| Task | Guide |
|---|---|
| Understand the dataset and fun labels | [Dataset overview](datasets/fun-study-dataset-analysis/dataset-explained.md) |
| Prepare training data | [Preprocessing](datasets/fun-study-dataset-analysis/PREPROCESSING.md) |
| Visualize encoded levels and paths | [Level rendering](datasets/fun-study-dataset-analysis/ENCODING_TO_LEVEL_IMAGE.md) |
| Review baseline measurements | [Baseline metrics](datasets/fun-study-dataset-analysis/baseline-metrics.md) |
| Set up the environment and choose a model | [Training guide and architecture index](model-training/TRAINING.md) |
| Compare completed experiments | [Saved-run analysis](model-training/ANALYZE_RUNS.md) |
| Reconstruct inputs or generate new levels | [Reconstruction and generation](model-training/RECONSTRUCTION_AND_GENERATION.md) |
| Read the research plan | [Project proposal](model-training/player-exp-pairwise-cnn-vae-pcgml-proposal.pdf) |
