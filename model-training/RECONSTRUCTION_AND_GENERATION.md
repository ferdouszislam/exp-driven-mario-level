# Run reconstruction and generation after training

Use [generate_levels.py](generate_levels.py) after training has finished and the
model checkpoints have been saved. It automatically loads the matching model
architecture for each checkpoint.

For every checkpoint, it creates:

- Reconstructions of five distinct levels from the fitting portion of the training set.
- Reconstructions of five distinct levels from the recorded validation portion.
- Five new levels from standard-normal latent vectors, `z ~ N(0, I)`, passed directly to the decoder.

If a fitting or validation portion contains fewer than five distinct levels, it
uses all available levels. If validation is empty, its manifest explains the skip.
The five new decoder samples are still generated.

## Before running

You need the repository's root `.venv`, the preprocessed dataset, and saved `.pt`
files inside `checkpoints/` folders. Training saves these files only for the
configured checkpoint ranges. See [TRAINING.md](TRAINING.md) for training and
checkpoint-saving instructions.

Run the commands below from the **repository root**, the directory containing
`.venv/`, `model-training/`, and `datasets/`.

## Process all saved checkpoints

```bash
.venv/bin/python model-training/generate_levels.py
```

This searches every `checkpoints/` folder beneath `model-training/`, including
archived runs, and processes all saved `.pt` files. You do not need to supply the
architecture name or the checkpoint epoch ranges again.

If your terminal is already inside `model-training/`, the equivalent command is:

```bash
../.venv/bin/python generate_levels.py
```

## Process a smaller selection

Use `--runs-dir` to search only one architecture's runs:

```bash
.venv/bin/python model-training/generate_levels.py \
  --runs-dir model-training/model_arch_1/runs
```

You can also point `--runs-dir` to one completed run. Replace `my_run` with your
actual run folder:

```bash
.venv/bin/python model-training/generate_levels.py \
  --runs-dir model-training/model_arch_1/runs/my_run
```

Use `--checkpoints-dir` to process only one hyperparameter combination's saved
models. Replace both the run and parameter folder names with your actual paths:

```bash
.venv/bin/python model-training/generate_levels.py \
  --checkpoints-dir model-training/model_arch_1/runs/my_run/1e-4_1_1e-3_1e-2/checkpoints
```

`--runs-dir` and `--checkpoints-dir` cannot be used together.
Paths supplied to these flags, `--dataset`, or `--output-dir` are relative to your terminal's
current directory unless you provide an absolute path.

## Choose an output directory

Pass `--output-dir` to save results at another location. With `--checkpoints-dir`,
the requested directory directly contains `train/`, `validation/`, `generated/`,
and `index.json`:

```bash
.venv/bin/python model-training/generate_levels.py \
  --checkpoints-dir model-training/model_arch_1/runs/my_run/1e-4_1_1e-3_1e-2/checkpoints \
  --output-dir generated-levels
```

For recursive searches, each combination's path relative to `--runs-dir` (or
`model-training/` by default) is retained beneath the output directory. This
keeps checkpoints with matching epoch names in separate folders. For example,
`--runs-dir my_run --output-dir generated-levels` saves a combination's results
under `generated-levels/parameter_folder/`. Pointing either search flag directly
at a `checkpoints/` folder writes results directly into the output directory.
Omit `--output-dir` to keep the existing locations beside each checkpoint folder.

## Use a dataset at a different location

The script first uses the dataset path recorded in each checkpoint. If that path
is unavailable, it tries the local
`datasets/fun-study-dataset-analysis/preprocessed_dataset/` directory.

To choose the dataset explicitly, for example after downloading checkpoints from
a cluster, pass its preprocessed root with `--dataset`:

```bash
.venv/bin/python model-training/generate_levels.py \
  --runs-dir model-training/model_arch_1/runs/my_run \
  --dataset datasets/fun-study-dataset-analysis/preprocessed_dataset \
  --device cpu
```

The dataset must match the saved checkpoint's channel metadata, selected
comparisons, and fingerprint when recorded. The saved fitting and validation
memberships are reused; the script does not make a new split.

## Options

| Option | Meaning | Default |
|---|---|---|
| `--runs-dir PATH` | Recursively search a particular run root. | The script's `model-training/` directory. |
| `--checkpoints-dir PATH` | Process the `.pt` files directly inside one checkpoint folder. | Unset. |
| `--output-dir PATH` | Choose the output root, retaining relative combination folders for recursive searches. | `reconstructions/` beside each checkpoint folder. |
| `--dataset PATH` | Override the dataset location. | Saved path, then the local preprocessed dataset if that path is unavailable. |
| `--device auto`, `cpu`, `cuda`, or `mps` | Choose the inference device. | `auto`: CUDA, then MPS, then CPU, depending on availability. |
| `--cpu-threads N` | Set a positive number of CPU threads. | `4`. |
| `--seed N` | Choose another reproducible set of input levels and latent samples. Valid range: `0` through `4294967295`. | Each checkpoint's training seed. |
| `--help` | Show command-line help without generating anything. | — |

For example, add `--seed 123` to a command to change the sampled inputs and
decoder latent vectors. Keep the same seed to compare checkpoints using the same
inputs and latent noise when their saved splits match. This changes the sampled
visualization inputs, without changing the checkpoint's saved training/validation split.

## Find the results

By default, outputs are saved beside each combination's `checkpoints/` folder. For example:

```text
my_run/parameter_folder/
├── checkpoints/
│   ├── epoch_030.pt
│   └── epoch_031.pt
└── reconstructions/
    ├── index.json
    ├── train/
    │   ├── epoch_030/
    │   └── epoch_031/
    ├── validation/
    │   ├── epoch_030/
    │   └── epoch_031/
    └── generated/
        ├── epoch_030/
        └── epoch_031/
```

Open `comparison.png` inside an epoch folder to see all of that category's
results. Training and validation sheets pair each original input with its
reconstruction; generated sheets show the five decoder samples.

| File inside an epoch folder | Contents |
|---|---|
| `01_input.png` through `05_input.png` | Original inputs; present for training and validation reconstructions. |
| `01_generated.png` through `05_generated.png` | Reconstructed or newly sampled level images. |
| `01_generated.npy` through `05_generated.npy` | Binary `(22, 16, 176)` level encodings. |
| `01_latent.npy` through `05_latent.npy` | Sampled latent vectors; present in `generated/` folders. |
| `comparison.png` | Combined, labeled image sheet. |
| `samples.json` | Checkpoint, epoch, seeds, output filenames, and input provenance where applicable. |
| `channels.json` | The renderer's channel codebook. |

Smaller input sets have fewer numbered files. An empty validation folder contains
only `samples.json`. Each output directory's `index.json` lists the
checkpoints and their output paths, and the terminal prints each saved sheet path.

## Reruns and errors

Rerunning with the same seed, dataset, model, and execution environment reproduces
the outputs. Existing results for each processed checkpoint are replaced only
after its new outputs have been generated successfully. Other checkpoints and
the saved training results stay in place.

The terminal ends with `Completed X/Y checkpoints`. If a checkpoint fails, the
script prints its path and error, continues with the remaining checkpoints, and
exits with a nonzero status. Check these common causes:

- **No saved checkpoints:** check the search path and confirm training saved `.pt` files within the configured ranges.
- **Dataset mismatch or missing comparisons:** use the matching preprocessed dataset with `--dataset`.
- **Architecture or model-state mismatch:** use the model definitions that match the saved checkpoint.
- **Requested device unavailable:** select `--device cpu` or an available accelerator.
