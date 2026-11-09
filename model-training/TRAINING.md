# CNN-VAE and pairwise fun-ranking training

This pipeline implements the joint training stage of the project proposal using the specified layer dimensions. It trains one shared CNN-VAE and a fully connected classifier on two concatenated latent samples. The classifier predicts whether the first playthrough is preferred to the second. [Project proposal](player-exp-pairwise-cnn-vae-pcgml-proposal.pdf), [architecture specification](initial_model_architecture.md).

The current workflow trains **12 independent hyperparameter combinations for 100 epochs each**. Every epoch records fitting and validation measurements, and each combination produces loss curves. You choose a combination and epoch from those records. Each experiment starts with a fresh model and uses the same grouped split drawn from the supplied training portion. The modules cover data loading, the model, experiment execution, and plotting. [data.py](data.py), [model.py](model.py), [train.py](train.py), [plotting.py](plotting.py).

## 1. Setup and commands

Run these commands from the repository root. The training dependencies are pinned in [requirements.txt](requirements.txt). Reuse a root `.venv` if it exists:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r model-training/requirements.txt
```

The installed environment was checked with Python **3.14.7**, PyTorch **2.15.0**, NumPy **2.3.4**, and Matplotlib **3.11.2**. Matplotlib generates the saved loss curves.

For a small end-to-end run with four original comparisons and two epochs per combination, use

```bash
.venv/bin/python model-training/train.py \
  --num-training-samples 4 \
  --epochs 2 \
  --output-dir model-training/runs/comparison_smoke
```

For all 12 combinations using the complete supplied training portion and the 100-epoch default, use

```bash
.venv/bin/python model-training/train.py \
  --output-dir model-training/runs/comparison_100_epochs
```

The checkpoint-saving boundary can be exercised with four comparisons and 21 epochs per combination

```bash
.venv/bin/python model-training/train.py \
  --num-training-samples 4 \
  --epochs 21 \
  --output-dir model-training/runs/comparison_checkpoint_check
```

That run saves one epoch-21 checkpoint per combination. Runs of 20 epochs or fewer produce epoch measurements and plots, with their checkpoint directories containing zero checkpoints.

Each output directory must be new or empty. Omit `--output-dir` to create a new UTC-timestamped directory under `model-training/runs/`. The script rejects output inside the input dataset. View all flags with

```bash
.venv/bin/python model-training/train.py --help
```

## 2. Inputs and training-only data access

The default dataset root is:

```text
datasets/fun-study-dataset-analysis/preprocessed_dataset/
```

The loader reads `channels.json` and the global `manifest.json`, filters manifest entries to `split == "train_set"`, and opens only their `train_set/` comparison files. It does not open test-set rankings or tensors. The default input location is resolved relative to `train.py`, so it also works when the script is launched from another working directory. `--dataset` can select a compatible preprocessed root containing `train_set/`, `channels.json`, and schema-version-2 `manifest.json`.

The separate `encoding_level_images/train_set/` directory contains three rendered PNG illustrations. The network uses the saved multi-hot `.npy` tensors in `preprocessed_dataset/train_set/`. PNG pixel colors and black path-marker strokes are not model inputs. See [PREPROCESSING.md](../datasets/fun-study-dataset-analysis/PREPROCESSING.md) for how tensors and labels were formed.

The supplied train portion has **290 original comparisons**, comprising 77 Experiment I comparisons and 213 pairs derived from Experiment II rankings. It contains 367 distinct participant-level playthroughs. Every comparison folder stores its own two `.npy` inputs and `ranking.txt`; the first line of that ranking file identifies the preferred level. A participant's path remains attached to its own level observation. Repeated level IDs are not merged across participants.

Each tensor has shape **`(22, 16, 192)`**, channels first, and preserves all 192 stored columns. Loading converts binary `uint8` values to `float32` values 0 or 1 without rescaling. There is no crop, interpolation, normalization of pixel intensities, or image augmentation.

Channels 0–20 have these literal tile symbols, in order:

```text
X S - % t T o Q @ U L 1 2 g k r K R y # |
```

Channel 21 is `PATH`, represented by `P` in path text. Exactly one of the first 21 bits is active at every cell. A visited cell also has its independent path bit active, so tile and path can coexist. The saved codebook is checked against this order, shape, axis order, and path-channel index.

The loader checks train-folder inventory against the train manifest entries, verifies ranking/level-ID agreement, and validates **all training arrays** for shape, dtype, binary values, and one active geometry bit. This validation also runs when a low sample cap is used. Only selected arrays are cached as float32 tensors. File paths must remain inside `train_set/`; paths that resolve outside it are rejected.

## 3. Setting the number of training samples

Near the top of `train.py`, set:

```python
NUM_TRAINING_SAMPLES: int | None = None
```

`None` uses all 290 train comparisons. Set a positive integer, such as `4`, for a smaller run. `--num-training-samples 4` overrides the variable for one invocation. The number counts **original comparison folders before reversing their order**, rather than individual playthroughs or optimization batches. A request larger than the available comparison count is rejected.

Selection preserves entire `split_group_id` groups. It sorts group IDs, shuffles them with `random.Random(seed)`, and greedily includes each whole group if it fits within the cap. Selected entries are then sorted by instance ID. The actual selected count can therefore fall below the requested maximum; it never exceeds the cap. A cap that cannot fit any complete group fails explicitly. The current training portion has 148 groups: 77 single-comparison groups and 71 three-comparison groups.

This rule keeps all pairs from the same ranking and all other linked observations together. The same grouping also controls the fixed fitting/validation split shared by all 12 combinations. `run_config.json` records requested and actual comparison counts, and `selected_train_comparisons.json` records the complete selected entries.

## 4. Ordered inputs and targets

For every retained comparison `preferred > other`, training presents both views:

| Input A | Input B | Target y |
|---|---|---:|
| Preferred playthrough | Other playthrough | 1 |
| Other playthrough | Preferred playthrough | 0 |

Thus the complete train portion yields **580 ordered examples** with 290 targets of each class. The default internal split assigns 464 ordered examples to fitting and 116 to validation. Reversal occurs after sample selection and after the fitting/validation split, so both orientations remain in the same group and partition. Fitting batches are shuffled each epoch, using a seeded DataLoader with `num_workers=0`.

Each original comparison has equal weight. There is no extra source-annotation normalization, duplicate removal, class weighting, or resampling. Consequently, a three-pair annotation contributes three original comparisons, exactly as stored in the prepared dataset. Reversing every pair doubles all comparison contributions equally. Repeated appearances of a playthrough remain part of their respective comparison losses.

## 5. Shared model architecture

Both inputs pass independently through **the same encoder parameters**. Both latent vectors use **the same decoder parameters**, and all ordered pairs use **the same FC-Tail**. The implementation creates one encoder, one decoder, and one pair classifier.

The encoder applies four `Conv2d` layers, each with a 4×4 kernel, stride 2, and padding 1:

| Layer | Channels | Output shape per playthrough |
|---|---|---|
| Input | 22 | 22 × 16 × 192 |
| Conv 1 | 22 → 32 | 32 × 8 × 96 |
| Conv 2 | 32 → 64 | 64 × 4 × 48 |
| Conv 3 | 64 → 128 | 128 × 2 × 24 |
| Conv 4 | 128 → 128 | 128 × 1 × 12 |
| Flatten and hidden FC | 1,536 → 128 | 128 |
| Separate mean and log-variance heads | 128 → 32 each | μ: 32; log_var: 32 |

Every encoder convolution is followed by GroupNorm, then LeakyReLU with negative slope 0.2. The hidden FC is followed by the same activation. GroupNorm uses **8 groups**, a valid divisor of every normalized channel count; the architecture document leaves this group count open. There is no pooling.

The diagonal Gaussian posterior is sampled by reparameterization:

```text
z = μ + exp(0.5 × log_var) × ε,   ε ~ N(0, I)
```

Each playthrough has a 32-dimensional latent sample. Sampling is used in both training and reported loss/metric computation.

The decoder projects `32 → 1,536`, reshapes to `(128, 1, 12)`, and applies four `ConvTranspose2d` layers, also with kernel 4×4, stride 2, and padding 1:

```text
128 × 1 × 12
  → 128 × 2 × 24
  →  64 × 4 × 48
  →  32 × 8 × 96
  →  22 × 16 × 192
```

The first three transposed convolutions use GroupNorm with 8 groups followed by LeakyReLU(0.2). The final layer returns unrestricted logits, one per input bit. There is no final sigmoid inside the decoder and no unpooling.

The FC-Tail concatenates samples **in the supplied input order**:

```text
[z_A; z_B]: 64
  → Linear(64,64) → LeakyReLU(0.2) → Dropout(0.1)
  → Linear(64,32) → LeakyReLU(0.2)
  → Linear(32,1)
```

The last scalar is a logit. Applying sigmoid gives `P(first playthrough preferred)`. Returning logits permits numerically stable BCE-with-logits loss, as allowed in the architecture specification. The network does not impose a scalar utility function or an exact complement relationship between its two input orders.

### Parameter count

The implemented model has **1,138,199 parameters**, all trainable. Counting `sum(parameter.numel() for parameter in model.parameters())` gives the following breakdown. The shared encoder and decoder are each counted once, even though each processes both playthroughs in a pair.

| Component | Trainable parameters |
|---|---:|
| Shared encoder, including mean and log-variance heads | 643,296 |
| Shared decoder | 488,630 |
| Pairwise FC-Tail | 6,273 |
| **Total** | **1,138,199** |

The layer counts include convolutional and fully connected weights and biases. Each GroupNorm contributes one learned scale and one learned shift per output channel, giving `2 × channels` parameters. Rows containing GroupNorm include these parameters in their totals.

| Component | Layer | Parameters |
|---|---|---:|
| Encoder | Conv 22 → 32, 4×4, followed by GroupNorm | 11,360 |
| Encoder | Conv 32 → 64, 4×4, followed by GroupNorm | 32,960 |
| Encoder | Conv 64 → 128, 4×4, followed by GroupNorm | 131,456 |
| Encoder | Conv 128 → 128, 4×4, followed by GroupNorm | 262,528 |
| Encoder | Hidden FC 1,536 → 128 | 196,736 |
| Encoder | Mean head 128 → 32 | 4,128 |
| Encoder | Log-variance head 128 → 32 | 4,128 |
| Decoder | Projection FC 32 → 1,536 | 50,688 |
| Decoder | Transposed conv 128 → 128, 4×4, followed by GroupNorm | 262,528 |
| Decoder | Transposed conv 128 → 64, 4×4, followed by GroupNorm | 131,264 |
| Decoder | Transposed conv 64 → 32, 4×4, followed by GroupNorm | 32,864 |
| Decoder | Output transposed conv 32 → 22, 4×4 | 11,286 |
| FC-Tail | FC 64 → 64 | 4,160 |
| FC-Tail | FC 64 → 32 | 2,080 |
| FC-Tail | FC 32 → 1 | 33 |

Flattening, reshaping, concatenation, LeakyReLU, dropout, sigmoid, and latent sampling each contribute zero parameters.

## 6. Joint losses and their reductions

The optimizer minimizes:

```text
L_total = λ_rec × L_rec + λ_KL × L_KL + λ_rank × L_rank
```

For a batch containing `B` ordered examples:

1. **Reconstruction BCE:** compute binary cross-entropy with logits independently for A and B. Each side is averaged over `B × 22 × 16 × 192` bits, then the two side losses are averaged. All geometry and path bits have equal weight. The loss remains binary cross-entropy across all channels, as specified in the proposal.
2. **KL divergence:** for each side and each example, compute `-0.5 × sum_j(1 + log_var_j - μ_j² - exp(log_var_j))`, summing over all 32 latent coordinates. Average over the batch, then average the A and B results. This measures divergence from the standard normal prior; it is not divided by 32.
3. **Ranking BCE:** compute binary cross-entropy with logits between the FC-Tail output and the ordered-pair target, averaged over the batch.

Training reports each unweighted component and their weighted total. Epoch losses average batch values using the number of ordered examples in each batch, so a shorter last batch receives its correct weight. Dropout and fresh posterior draws are active during optimization. Ranking gradients flow through both sampled latent vectors into the shared encoder and FC-Tail. Reconstruction gradients train the encoder and decoder, while the KL term trains the posterior parameters through the encoder.

PyTorch documents the fused loss and its mean reduction in [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.15/generated/torch.nn.BCEWithLogitsLoss.html).

## 7. Independent experiments and the fixed split

The experiment grid combines two Adam learning rates, two KL weights, and three reconstruction/ranking weight pairs. Each invocation runs all **12 combinations**, ordered first by learning rate, then KL weight, then reconstruction/ranking pair.

| Directory | Adam learning rate | λ_rec | λ_KL | λ_rank |
|---|---:|---:|---:|---:|
| `combination_01` | 0.001 | 1 | 0.001 | 1 |
| `combination_02` | 0.001 | 2 | 0.001 | 1 |
| `combination_03` | 0.001 | 1 | 0.001 | 2 |
| `combination_04` | 0.001 | 1 | 0.01 | 1 |
| `combination_05` | 0.001 | 2 | 0.01 | 1 |
| `combination_06` | 0.001 | 1 | 0.01 | 2 |
| `combination_07` | 0.0003 | 1 | 0.001 | 1 |
| `combination_08` | 0.0003 | 2 | 0.001 | 1 |
| `combination_09` | 0.0003 | 1 | 0.001 | 2 |
| `combination_10` | 0.0003 | 1 | 0.01 | 1 |
| `combination_11` | 0.0003 | 2 | 0.01 | 1 |
| `combination_12` | 0.0003 | 1 | 0.01 | 2 |

Each combination receives a freshly initialized model and Adam optimizer. Resetting the configured seed, 42 by default, before each experiment gives the same initial model weights and fitting-data order across combinations. Adam uses its PyTorch defaults apart from the specified learning rate. The combination keeps its own learning rate and loss weights throughout its epoch budget.

A single internal fitting/validation split is created from the selected `train_set` comparisons and reused across all experiments. The default `--validation-fraction 0.2` targets 20% of the original comparisons for validation. Group IDs are sorted and seeded-shuffled, then whole groups are assigned until the target is reached while retaining at least one group for fitting. With all 290 supplied comparisons and seed 42, this gives **232 fitting comparisons and 58 validation comparisons**. Both input orders are then generated inside their assigned portion.

Each combination fits on those 232 comparisons for its entire **100-epoch** default budget. The fixed validation portion supports comparison of epochs and experiments. After all experiments finish, you choose the combination and epoch using the recorded measurements and curves. Every combination completes the requested epoch budget independently.

For a different subset or seed, indivisible group sizes can change the achieved validation fraction. If only one selected group is available, the fitting portion contains all selected comparisons and validation measurements are recorded as `null`.

The main controls are available near the top of `train.py` and through the CLI.

| Control | Default | Meaning |
|---|---:|---|
| `NUM_TRAINING_SAMPLES` / `--num-training-samples` | `None` | Cap on original comparisons, preserving whole groups |
| `EPOCHS` / `--epochs` | 100 | Full epoch budget for each of the 12 combinations |
| `--batch-size` | 16 | Ordered fitting examples per optimization batch |
| `--validation-fraction` | 0.2 | Target fraction of selected comparisons for fixed validation |
| `--seed` | 42 | Initialization, sample selection, split, and random-stream seed |
| `--device` | `auto` | CUDA when available, otherwise MPS when available, otherwise CPU |
| `--cpu-threads` | 4 | PyTorch CPU thread count |
| `--output-dir` | New timestamped directory | Root directory for all 12 experiments |

`--dataset` can select a compatible preprocessed root. The default dataset path is resolved relative to the script. An explicitly requested unavailable accelerator raises an error.

## 8. Measurements after every epoch

Each epoch first performs its optimization updates with dropout and fresh posterior samples active. The optimizer sees fitting comparisons. At the end of that epoch, the fixed model is measured separately on the complete fitting portion and the complete validation portion.

These two measurements use `model.eval()`, disabled gradients, and dropout off. Both use the reporting seed `(seed + 1) % (2**32)`, which is 43 by default. Reusing that seed at every epoch applies the same random noise sequence to the same input occurrences as model parameters evolve. Each pair's two sampled latent vectors are reused for both input orders. A playthrough appearing in several comparison folders is sampled separately for each occurrence.

The evaluator saves and restores random-number-generator states around measurement. The next optimization epoch therefore continues its training random streams from their previous state. Measurement leaves model parameters fixed.

For each measured pair, reconstruction BCE and KL are calculated once and averaged over its two inputs. Ranking BCE averages the forward target-1 and reverse target-0 losses. Dataset losses then average over original comparisons. The total uses that combination's own weights and the reductions defined in section 6.

Every epoch prints fitting and validation values to the console and writes them to JSON and CSV. The recorded measurements include

- Weighted total loss, reconstruction BCE, KL divergence, and ranking BCE.
- Overall ordered-pair accuracy.
- Accuracy within true class 0 and within true class 1.
- Binary F1-score for positive class 1.

Classification uses `sigmoid(logit) >= 0.5` to predict class 1 and assigns class 0 below that threshold. Class 1 means the first playthrough is preferred. Class 0 means the second playthrough is preferred. Both input orders contribute to the measurements. The confusion matrix has true-label rows and predicted-label columns ordered `[0, 1]`.

```text
accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / ordered_examples
accuracy_class_0 = confusion_matrix[0][0] / class_support["0"]
accuracy_class_1 = confusion_matrix[1][1] / class_support["1"]

TP = confusion_matrix[1][1]
FP = confusion_matrix[0][1]
FN = confusion_matrix[1][0]
f1_score = 2 × TP / (2 × TP + FP + FN)
```

Each class-specific accuracy is the fraction of correctly classified examples among examples whose true label is that class. Saved metrics expose these values through `accuracy_by_class`. Binary F1 uses `f1_score` and `f1_positive_label=1`. Per-class F1 diagnostics, class supports, and the confusion matrix are also saved. A zero denominator gives a value of 0.

Epoch history also records `total_loss_gap = abs(validation_total_loss - train_total_loss)` when validation is available. This value accompanies the curves for manual comparison of combinations and epochs. The gap is `null` for a single-group run.

The optimization loss averaged during an epoch can differ from that epoch's fitting measurement because optimization uses changing weights, active dropout, and fresh posterior draws. The fitting measurement uses the fixed model at the end of the epoch.

Python, NumPy, PyTorch, accelerator RNGs, and fitting-data shuffling are seeded. cuDNN benchmarking is disabled and its deterministic option is enabled when available. Numerical results can vary across devices, PyTorch releases, or batch sizes. The run records its environment and reporting convention. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.15/notes/randomness.html).

## 9. Saved outputs, curves, and checkpoints

Each invocation creates one experiment root with the shared configuration and split.

| File | Contents |
|---|---|
| `run_config.json` | Dataset and fingerprint, requested/actual sample counts, epoch budget, seed, device, environment, architecture, and shared run settings |
| `channels.json` | Exact input channel codebook |
| `selected_train_comparisons.json` | Selected manifest entries with participant, playthrough, and grouping provenance |
| `split.json` | Fixed fitting and validation memberships shared by all combinations |
| `experiments.json` | The 12 combinations, their `pending`, `running`, `completed`, or `failed` status, and completed experiment summaries |
| `REPORT.md` | Human-readable run settings and links to experiment outputs after all combinations finish |

Each `combination_01/` through `combination_12/` directory contains its own outputs.

| File | Contents |
|---|---|
| `config.json` | This combination's hyperparameters and experiment settings |
| `train_history.json` | One row per epoch containing `epoch`, optimization losses, full `train` metrics, `validation` metrics or `null`, `total_loss_gap` or `null`, and duration in seconds |
| `epoch_metrics.csv` | Flattened per-epoch measurements using `optimization_`, `train_`, and `validation_` column prefixes, plus duration and the absolute total-loss gap |
| `train_metrics.json` | Complete fitting metrics at the final epoch |
| `validation_metrics.json` | Complete validation metrics at the final epoch, or JSON `null` for a single-group run |
| `train_predictions.csv` | Final-epoch ordered predictions for the fitting portion |
| `validation_predictions.csv` | Final-epoch ordered validation predictions when validation is available |
| `loss_curves.png` | A 2×2 figure for weighted total loss, reconstruction BCE, KL divergence, and ranking BCE, each showing fitting and validation values against epoch number |
| `checkpoints/epoch_021.pt`, etc. | Model and optimizer state for each completed epoch strictly greater than 20 |

The JSON history and complete epoch CSV are each replaced atomically after every epoch's measurements. This makes completed epoch measurements available while the experiment continues. Prediction CSV files summarize the final epoch. The history retains measurements for earlier epochs.

The dataset SHA-256 fingerprint includes the codebook bytes and selected training-entry metadata, ranking bytes, and encoding bytes, with explicit boundaries between components. Its inputs are the selected training observations.

Checkpoint saving starts at **epoch 21**. With 100 epochs, each combination produces checkpoints for epochs **21–100 inclusive**, giving **80 per combination and 960 across all 12 combinations**. A checkpoint containing the 1,138,199-parameter model and Adam state is approximately **13.7 MB**. The complete default checkpoint collection therefore uses roughly **13 GB**, plus reports and plots. A run ending at epoch 20 or earlier has zero checkpoints.

Each checkpoint contains `epoch`, `model_state_dict`, `optimizer_state_dict`, `config`, `channels`, `selected_instance_ids`, `fitting_instance_ids`, `validation_instance_ids`, `train_metrics`, and `validation_metrics`. The configuration and split memberships identify the experiment and data associated with that saved epoch.

To inspect a checkpoint you have generated, choose its combination and epoch explicitly. This example loads combination 1 at epoch 21 from the full-run command in section 1.

```python
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path("model-training").resolve()))
from model import PairwiseCNNVAE

checkpoint = torch.load(
    "model-training/runs/comparison_100_epochs/combination_01/checkpoints/epoch_021.pt",
    map_location="cpu",
    weights_only=True,
)
assert checkpoint["epoch"] == 21
assert checkpoint["config"]["combination_id"] == "combination_01"
assert checkpoint["config"]["hyperparameters"] == {
    "learning_rate": 0.001,
    "lambda_rec": 1.0,
    "lambda_kl": 0.001,
    "lambda_rank": 1.0,
}
print(checkpoint["config"])
print(checkpoint["fitting_instance_ids"])
print(checkpoint["validation_instance_ids"])

model = PairwiseCNNVAE()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(checkpoint["train_metrics"])
print(checkpoint["validation_metrics"])
```

The assertions verify the saved epoch, combination ID, and hyperparameters against the chosen grid row. `map_location="cpu"` maps the saved model and optimizer tensors to CPU during loading.

## 10. Verification and existing artifacts

Run the behavioral checks from the repository root.

```bash
.venv/bin/python -m unittest discover -s model-training/tests -v
```

The two-epoch command in section 1 exercises all 12 independent experiments, per-epoch measurements, JSON/CSV export, and curves. The 21-epoch command additionally exercises the checkpoint boundary. Use a fresh output directory for each invocation.

Existing directories under `model-training/runs/` contain artifacts from earlier training workflows. Their saved configurations and reports identify the procedure and metric schema used for those runs. The experiment layout, commands, and checkpoint policy above describe newly generated runs under the current workflow.
