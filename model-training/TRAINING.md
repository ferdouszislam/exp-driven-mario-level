# CNN-VAE and pairwise fun-ranking training

This pipeline implements the joint training stage of the project proposal using the specified layer dimensions. It trains one shared CNN-VAE and a fully connected classifier on two concatenated latent samples. The classifier predicts whether the first playthrough is preferred to the second. [Project proposal](player-exp-pairwise-cnn-vae-pcgml-proposal.pdf), [current architecture specification](model_arch_1.md).

The current workflow trains **16 independent hyperparameter combinations for 50 epochs each**. Every epoch records fitting and validation measurements, and each combination produces curves for losses, overall accuracy, and normalized tile Hamming distance. You choose a combination and epoch from those records. Each experiment starts with a fresh model and uses the same saved `train_set` and `validation_set` assignments. Checkpoint saving is enabled by supplying both epoch-range bounds. The modules cover data loading, the model, experiment execution, and plotting. [data.py](data.py), [model.py](model.py), [train.py](train.py), [plotting.py](plotting.py).

## 1. Setup and commands

Run these commands from the repository root. The training dependencies are pinned in [requirements.txt](requirements.txt). Reuse a root `.venv` if it exists:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r model-training/requirements.txt
```

The installed environment was checked with Python **3.14.7**, PyTorch **2.15.0**, NumPy **2.3.4**, and Matplotlib **3.11.2**. Matplotlib generates each combination's saved `validation_curves.png`.

For a small end-to-end run with four original comparisons and two epochs per combination, use

```bash
.venv/bin/python model-training/train.py \
  --num-training-samples 4 \
  --epochs 2 \
  --output-dir model-training/runs/comparison_smoke
```

For all 16 combinations using the complete supplied training portion and the 50-epoch default, use

```bash
.venv/bin/python model-training/train.py \
  --output-dir model-training/runs/comparison_50_epochs
```

This command records measurements and curves with checkpoint saving disabled.

An explicit single-epoch checkpoint range can be exercised with four comparisons and 21 training epochs per combination

```bash
.venv/bin/python model-training/train.py \
  --num-training-samples 4 \
  --epochs 21 \
  --checkpoint-start-epoch 21 \
  --checkpoint-end-epoch 21 \
  --output-dir model-training/runs/comparison_checkpoint_check
```

That run saves one epoch-21 checkpoint per combination.

To train for the default 50 epochs and save epochs 30 through 40 inclusive, use

```bash
.venv/bin/python model-training/train.py \
  --checkpoint-start-epoch 30 \
  --checkpoint-end-epoch 40 \
  --output-dir model-training/runs/comparison_30_to_40
```

This saves 11 checkpoints per combination, giving 176 across all 16 combinations. The window controls saving while each experiment completes its full training budget.

Each output directory must be new or empty. Omit `--output-dir` to create a new UTC-timestamped directory under `model-training/runs/`. The script rejects output inside the input dataset. View all flags with

```bash
.venv/bin/python model-training/train.py --help
```

## 2. Inputs and training/validation data access

The default dataset root is:

```text
datasets/fun-study-dataset-analysis/preprocessed_dataset/
```

The loader reads `channels.json` and the global `manifest.json`, then opens comparisons assigned to `train_set` and, when present, `validation_set`. It does not open test-set rankings or tensors. The default input location is resolved relative to `train.py`, so it also works when the script is launched from another working directory. `--dataset` can select a compatible preprocessed root containing `train_set/`, optional `validation_set/`, `channels.json`, and schema-version-2 `manifest.json`.

The separate `encoding_level_images/train_set/` directory contains three rendered PNG illustrations. The network uses the saved multi-hot `.npy` tensors in `preprocessed_dataset/train_set/`. PNG pixel colors and black path-marker strokes are not model inputs. See [PREPROCESSING.md](../datasets/fun-study-dataset-analysis/PREPROCESSING.md) for how tensors and labels were formed.

The supplied `train_set` has **232 original comparisons**, comprising 67 Experiment I comparisons and 165 pairs derived from Experiment II rankings. The saved `validation_set` has **58 comparisons**: 10 from Experiment I and 48 from Experiment II. Together they contain the original 290 comparisons and 367 distinct participant-level playthroughs. The separate 73-comparison test set remains held out. Every comparison folder stores its own two `.npy` inputs and `ranking.txt`; the first line of that ranking file identifies the preferred level. A participant's path remains attached to its own level observation. Repeated level IDs are not merged across participants.

Each tensor has shape **`(22, 16, 192)`**, channels first, and preserves all 192 stored columns. Loading converts binary `uint8` values to `float32` values 0 or 1 without rescaling. There is no crop, interpolation, normalization of pixel intensities, or image augmentation.

Channels 0–20 have these literal tile symbols, in order:

```text
X S - % t T o Q @ U L 1 2 g k r K R y # |
```

Channel 21 is `PATH`, represented by `P` in path text. Exactly one of the first 21 bits is active at every cell. A visited cell also has its independent path bit active, so tile and path can coexist. The saved codebook is checked against this order, shape, axis order, and path-channel index.

The loader checks each train/validation folder inventory against its manifest entries, verifies ranking/level-ID agreement, and validates **all training and validation arrays** for shape, dtype, binary values, and one active geometry bit. This validation also runs when a low sample cap is used. Selected training arrays and all saved validation arrays are cached as float32 tensors. File paths must remain inside their assigned split directory. Shared `split_group_id` values or playthrough keys between training and saved validation are rejected.

## 3. Setting the number of training samples

Near the top of `train.py`, set:

```python
NUM_TRAINING_SAMPLES: int | None = None
```

`None` uses all 232 train comparisons. Set a positive integer, such as `4`, for a smaller run. `--num-training-samples 4` overrides the variable for one invocation. The number counts **original comparison folders before reversing their order**, rather than individual playthroughs or optimization batches. A request larger than the available comparison count is rejected. The cap applies only to `train_set`; all 58 saved validation comparisons are retained even for a small training run.

Selection preserves entire `split_group_id` groups. It sorts group IDs, shuffles them with `random.Random(seed)`, and greedily includes each whole group if it fits within the cap. Selected entries are then sorted by instance ID. The actual selected count can therefore fall below the requested maximum; it never exceeds the cap. A cap that cannot fit any complete group fails explicitly. The current training portion has 122 groups: 67 single-comparison groups and 55 three-comparison groups.

This rule keeps all pairs from the same ranking and all other linked observations together. The saved fitting/validation split uses the same grouping and is shared by all 16 combinations. `run_config.json` records requested and actual comparison counts, `selected_train_comparisons.json` records the complete selected training entries, and `selected_validation_comparisons.json` records the saved validation entries.

## 4. Ordered inputs and targets

For every retained comparison `preferred > other`, training presents both views:

| Input A | Input B | Target y |
|---|---|---:|
| Preferred playthrough | Other playthrough | 1 |
| Other playthrough | Preferred playthrough | 0 |

Thus the complete `train_set` yields **464 ordered examples** with 232 targets of each class. The saved validation portion yields 116 ordered examples with 58 targets of each class. Reversal occurs after sample selection and after the fitting/validation split, so both orientations remain in the same group and partition. Fitting batches are shuffled each epoch, using a seeded DataLoader with `num_workers=0`.

Each original comparison has equal weight. There is no extra source-annotation normalization, duplicate removal, class weighting, or resampling. Consequently, a three-pair annotation contributes three original comparisons, exactly as stored in the prepared dataset. Reversing every pair doubles all comparison contributions equally. Repeated appearances of a playthrough remain part of their respective comparison losses.

## 5. Shared model architecture

The current architecture is specified in `model_arch_1.md`. Both inputs pass independently through the same encoder parameters. Their sampled latent vectors use the same decoder parameters, and all ordered pairs use the same FC-Tail. [model_arch_1.md](model_arch_1.md).

The revision retains the original **4×4 kernels as an implementation assumption** and uses stride **2** with padding **1**. For these settings, each encoder convolution halves the spatial dimensions, and each transposed convolution doubles them.

| Encoder layer | Channels or FC dimensions | Output shape per playthrough | Following operations |
|---|---|---|---|
| Input | 22 channels | 22 × 16 × 192 | — |
| Conv 1 | 22 → 32, kernel 4×4, stride 2, padding 1 | 32 × 8 × 96 | LayerNorm((32, 8, 96)), ReLU |
| Conv 2 | 32 → 64, kernel 4×4, stride 2, padding 1 | 64 × 4 × 48 | LayerNorm((64, 4, 48)), ReLU |
| Conv 3 | 64 → 32, kernel 4×4, stride 2, padding 1 | 32 × 2 × 24 | LayerNorm((32, 2, 24)), ReLU |
| Flatten | 32 × 2 × 24 → 1,536 | 1,536 | — |
| Hidden FC | 1,536 → 128 | 128 | ReLU |
| Mean head | 128 → 32 | μ of shape 32 | Linear output |
| Log-variance head | 128 → 32 | log_var of shape 32 | Linear output |

All three encoder convolutions use LayerNorm followed by ReLU. LayerNorm normalizes over the complete `(channel, height, width)` feature map independently for each playthrough, with `eps=1e-5` and learned elementwise scale and bias. The encoder hidden FC uses ReLU, and the Gaussian heads return affine outputs. The diagonal Gaussian posterior is sampled by reparameterization. [PyTorch LayerNorm documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html).

```text
z = μ + exp(0.5 * log_var) * ε,   ε ~ N(0, I)
```

Each playthrough has a 32-dimensional latent sample. Sampling is used in both optimization and reported measurements.

| Decoder layer | Channels or FC dimensions | Output shape per playthrough | Following operations |
|---|---|---|---|
| Projection FC | 32 → 1,536 | 1,536 | ReLU |
| Reshape | 1,536 → 32 × 2 × 24 | 32 × 2 × 24 | — |
| Transposed conv 1 | 32 → 64, kernel 4×4, stride 2, padding 1 | 64 × 4 × 48 | LayerNorm((64, 4, 48)), ReLU |
| Transposed conv 2 | 64 → 32, kernel 4×4, stride 2, padding 1 | 32 × 8 × 96 | LayerNorm((32, 8, 96)), ReLU |
| Transposed conv 3 | 32 → 22, kernel 4×4, stride 2, padding 1 | 22 × 16 × 192 | Linear logits |

The decoder projection uses ReLU before reshaping. Its first two transposed convolutions use LayerNorm over their complete per-playthrough feature maps, followed by ReLU. The final transposed convolution returns unrestricted logits for all 22 input bits. The reconstruction loss applies BCE-with-logits to these outputs.

The FC-Tail concatenates samples in the supplied input order.

```text
[z_A; z_B] of size 64
  → Linear(64,64) → ReLU → Dropout(0.2)
  → Linear(64,32) → ReLU → Dropout(0.2)
  → Linear(32,1)
```

The final scalar is an affine logit. Sigmoid gives `P(first playthrough preferred)` for probabilities and classification. Both tail hidden layers use ReLU and dropout with probability 0.2.

### Parameter count

The implemented model has **566,327 parameters**, all trainable. Shared modules are counted once.

| Component | Trainable parameters |
|---|---:|
| Shared encoder, including mean and log-variance heads | 358,720 |
| Shared decoder | 201,334 |
| Pairwise FC-Tail | 6,273 |
| **Total** | **566,327** |

The layer counts include convolutional and fully connected weights and biases. LayerNorm contributes one learned scale and one learned shift per normalized feature-map element, giving `2 × C × H × W` parameters for each normalized shape.

| Component | Layer | Parameters |
|---|---|---:|
| Encoder | Conv 22 → 32, 4×4, followed by LayerNorm((32, 8, 96)) | 60,448 |
| Encoder | Conv 32 → 64, 4×4, followed by LayerNorm((64, 4, 48)) | 57,408 |
| Encoder | Conv 64 → 32, 4×4, followed by LayerNorm((32, 2, 24)) | 35,872 |
| Encoder | Hidden FC 1,536 → 128 | 196,736 |
| Encoder | Mean head 128 → 32 | 4,128 |
| Encoder | Log-variance head 128 → 32 | 4,128 |
| Decoder | Projection FC 32 → 1,536 | 50,688 |
| Decoder | Transposed conv 32 → 64, 4×4, followed by LayerNorm((64, 4, 48)) | 57,408 |
| Decoder | Transposed conv 64 → 32, 4×4, followed by LayerNorm((32, 8, 96)) | 81,952 |
| Decoder | Output transposed conv 32 → 22, 4×4 | 11,286 |
| FC-Tail | FC 64 → 64 | 4,160 |
| FC-Tail | FC 64 → 32 | 2,080 |
| FC-Tail | FC 32 → 1 | 33 |

Flattening, reshaping, concatenation, ReLU, dropout, sigmoid, and latent sampling each contribute zero parameters.

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

The experiment grid combines two Adam learning rates, two KL weights, and four reconstruction/ranking weight pairs. Each invocation runs all **16 combinations**, ordered first by learning rate, then KL weight, then reconstruction/ranking pair.

| Directory | Adam learning rate | λ_rec | λ_KL | λ_rank |
|---|---:|---:|---:|---:|
| `combination_01` | 0.0001 | 0.5 | 0.001 | 0.5 |
| `combination_02` | 0.0001 | 1 | 0.001 | 0.5 |
| `combination_03` | 0.0001 | 0.5 | 0.001 | 1 |
| `combination_04` | 0.0001 | 1 | 0.001 | 1 |
| `combination_05` | 0.0001 | 0.5 | 0.0001 | 0.5 |
| `combination_06` | 0.0001 | 1 | 0.0001 | 0.5 |
| `combination_07` | 0.0001 | 0.5 | 0.0001 | 1 |
| `combination_08` | 0.0001 | 1 | 0.0001 | 1 |
| `combination_09` | 0.00001 | 0.5 | 0.001 | 0.5 |
| `combination_10` | 0.00001 | 1 | 0.001 | 0.5 |
| `combination_11` | 0.00001 | 0.5 | 0.001 | 1 |
| `combination_12` | 0.00001 | 1 | 0.001 | 1 |
| `combination_13` | 0.00001 | 0.5 | 0.0001 | 0.5 |
| `combination_14` | 0.00001 | 1 | 0.0001 | 0.5 |
| `combination_15` | 0.00001 | 0.5 | 0.0001 | 1 |
| `combination_16` | 0.00001 | 1 | 0.0001 | 1 |

Each combination receives a freshly initialized model and Adam optimizer. Resetting the configured seed, 42 by default, before each experiment gives the same initial model weights and fitting-data order across combinations. Adam uses its PyTorch defaults apart from the specified learning rate. The combination keeps its own learning rate and loss weights throughout its epoch budget.

The saved `validation_set` is authoritative: fitting uses the selected `train_set` comparisons, and validation uses all saved validation comparisons across all experiments. Training does not split `train_set` a second time. The preprocessing default takes 20% of the original 290 training comparisons using whole groups and seed 42, giving **232 fitting comparisons and 58 validation comparisons**. Both input orders are then generated inside their assigned portion. `run_config.json` records `validation_source` as `validation_set` and sets its internal `validation_fraction` to `null` because no internal split was performed.

Each combination fits on those 232 comparisons for its entire **50-epoch** default budget. The fixed validation portion supports comparison of epochs and experiments. After all experiments finish, you choose the combination and epoch using the recorded measurements and curves. Every combination completes the requested epoch budget independently.

For legacy datasets without saved validation entries, training retains its internal split controlled by `--validation-fraction` (default `0.2`). It sorts and seeded-shuffles group IDs, then takes whole groups until reaching the target while retaining at least one fitting group. Indivisible groups can change the achieved fraction. If only one selected group is available in that legacy layout, validation measurements are recorded as `null`. In the saved layout, the training seed and sample cap do not change validation membership; change the preprocessing split to choose another validation set.

The main controls are available near the top of `train.py` and through the CLI.

| Control | Default | Meaning |
|---|---:|---|
| `NUM_TRAINING_SAMPLES` / `--num-training-samples` | `None` | Cap on original comparisons, preserving whole groups |
| `EPOCHS` / `--epochs` | 50 | Full epoch budget for each of the 16 combinations |
| `CHECKPOINT_START_EPOCH` / `--checkpoint-start-epoch` | `None` | Optional first checkpoint epoch, inclusive, supplied together with the end epoch |
| `CHECKPOINT_END_EPOCH` / `--checkpoint-end-epoch` | `None` | Optional last checkpoint epoch, inclusive, supplied together with the start epoch |
| `--batch-size` | 16 | Ordered fitting examples per optimization batch |
| `--validation-fraction` | 0.2 | Fallback target fraction for legacy datasets without saved validation entries |
| `--seed` | 42 | Initialization, sample selection, split, and random-stream seed |
| `--device` | `auto` | CUDA when available, otherwise MPS when available, otherwise CPU |
| `--cpu-threads` | 4 | PyTorch CPU thread count |
| `--output-dir` | New timestamped directory | Root directory for all 16 experiments |

`--dataset` can select a compatible preprocessed root. The default dataset path is resolved relative to the script. An explicitly requested unavailable accelerator raises an error.

## 8. Measurements after every epoch

Each epoch first performs its optimization updates with dropout and fresh posterior samples active. The optimizer sees fitting comparisons. At the end of that epoch, the fixed model is measured separately on the complete fitting portion and the complete validation portion.

These two measurements use `model.eval()`, disabled gradients, and dropout off. Both use the reporting seed `(seed + 1) % (2**32)`, which is 43 by default. Reusing that seed at every epoch applies the same random noise sequence to the same input occurrences as model parameters evolve. Each pair's two sampled latent vectors are reused for both input orders. A playthrough appearing in several comparison folders is sampled separately for each occurrence.

The evaluator saves and restores random-number-generator states around measurement. The next optimization epoch therefore continues its training random streams from their previous state. Measurement leaves model parameters fixed.

For each measured pair, reconstruction BCE and KL are calculated once and averaged over its two inputs. Ranking BCE averages the forward target-1 and reverse target-0 losses. Dataset losses then average over original comparisons. The total uses that combination's own weights and the reductions defined in section 6.

Every epoch prints fitting and validation values to the console and writes them to JSON and CSV. The recorded measurements include

- Weighted total loss, reconstruction BCE, KL divergence, and ranking BCE.
- Overall ordered-pair accuracy.
- Binary F1-score for positive class 1.
- Normalized tile Hamming distance between input and reconstructed geometry.

Classification uses `sigmoid(logit) >= 0.5` to predict class 1 and assigns class 0 below that threshold. Class 1 means the first playthrough is preferred. Class 0 means the second playthrough is preferred. Both input orders contribute to the measurements. The confusion matrix has true-label rows and predicted-label columns ordered `[0, 1]`.

```text
correct_ordered_predictions = confusion_matrix[0][0] + confusion_matrix[1][1]
accuracy = correct_ordered_predictions / ordered_examples

TP = confusion_matrix[1][1]
FP = confusion_matrix[0][1]
FN = confusion_matrix[1][0]
f1_score = 2 × TP / (2 × TP + FP + FN)
```

Binary F1 uses `f1_score` and `f1_positive_label=1`. Its value is 0 when the denominator is zero. Saved `class_support` and `confusion_matrix` diagnostics provide the underlying label and prediction counts.

Normalized Hamming distance measures reconstruction differences at the **tile level**. For each playthrough occurrence, the input tile ID is the argmax of its one-hot geometry channels 0–20. The reconstructed tile ID is the argmax of the corresponding decoder logits over those same 21 channels. The decoder reconstruction comes from the same current sampled latent used for the epoch's other measurements. Channel 21 records path occupancy and is excluded from this geometry comparison. Reconstruction BCE continues to use all 22 channels as defined in section 6.

Each playthrough contributes all **16 × 192 = 3,072 tile positions**. For a measured portion containing `N` original comparisons, the normalized distance aggregates both playthroughs in every comparison.

```text
input_tile_i[r,c] = argmax(input_i[0:21,r,c])
reconstructed_tile_i[r,c] = argmax(reconstruction_logits_i[0:21,r,c])

normalized_hamming_distance =
    sum over 2*N playthrough occurrences and all tile positions
        of 1[input_tile_i[r,c] != reconstructed_tile_i[r,c]]
    / (2 * N * 16 * 192)
```

The saved `normalized_hamming_distance` is a floating-point value from 0 to 1. Zero means every reconstructed tile matches its input tile, and 1 means every position differs. Each occurrence of a reused playthrough contributes to the aggregate. The metric is measured separately on fitting and validation portions after every epoch, appears as `Hamming` in console output, and is included in epoch history, CSV measurements, final metrics, and checkpoint metrics.

The optimization loss averaged during an epoch can differ from that epoch's fitting measurement because optimization uses changing weights, active dropout, and fresh posterior draws. The fitting measurement uses the fixed model at the end of the epoch.

Python, NumPy, PyTorch, accelerator RNGs, and fitting-data shuffling are seeded. cuDNN benchmarking is disabled and its deterministic option is enabled when available. Numerical results can vary across devices, PyTorch releases, or batch sizes. The run records its environment and reporting convention. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.15/notes/randomness.html).

## 9. Saved outputs, curves, and checkpoints

Each invocation creates one experiment root with the shared configuration and split.

| File | Contents |
|---|---|
| `run_config.json` | Dataset and fingerprint, requested/actual sample counts, epoch budget, checkpoint start/end epochs or two `null` values when disabled, seed, device, environment, architecture, and shared run settings |
| `channels.json` | Exact input channel codebook |
| `selected_train_comparisons.json` | Selected manifest entries with participant, playthrough, and grouping provenance |
| `selected_validation_comparisons.json` | Saved validation manifest entries when `validation_set` is present |
| `split.json` | Fixed fitting and validation memberships shared by all combinations |
| `experiments.json` | The 16 combinations, their `pending`, `running`, `completed`, or `failed` status, and completed experiment summaries, with `loss_graph` linking to each combination's `validation_curves.png` |
| `REPORT.md` | Human-readable run settings and links to each combination's six-panel fitting/validation curves after all combinations finish |

Each `combination_01/` through `combination_16/` directory contains its own outputs.

| File | Contents |
|---|---|
| `config.json` | This combination's hyperparameters and experiment settings |
| `train_history.json` | One row per epoch containing `epoch`, optimization losses, full `train` metrics, `validation` metrics or `null`, and duration in seconds |
| `epoch_metrics.csv` | Optimization losses with the `optimization_` prefix and post-epoch losses, overall accuracy, binary F1, and `normalized_hamming_distance` with `train_` and `validation_` prefixes, plus duration |
| `train_metrics.json` | Complete fitting metrics at the final epoch, including normalized tile Hamming distance |
| `validation_metrics.json` | Complete validation metrics at the final epoch, including normalized tile Hamming distance, or JSON `null` for a single-group run |
| `train_predictions.csv` | Final-epoch ordered predictions for the fitting portion |
| `validation_predictions.csv` | Final-epoch ordered validation predictions when validation is available |
| `validation_curves.png` | A 3×2 figure for weighted total loss, reconstruction BCE, KL divergence, ranking BCE, overall accuracy, and normalized tile Hamming distance, each showing fitting and available validation values against epoch number |
| `checkpoints/epoch_030.pt`, etc. | Optional model and optimizer state for each completed epoch within an explicitly configured inclusive checkpoint window |

The JSON history and complete epoch CSV are each replaced atomically after every epoch's measurements. This makes completed epoch measurements available while the experiment continues. Prediction CSV files summarize the final epoch. The history retains measurements for earlier epochs.

The overall-accuracy and normalized-Hamming-distance panels use a vertical range of 0–1. Higher accuracy indicates more correct preference predictions. Lower normalized Hamming distance indicates closer tile reconstruction. The filename `validation_curves.png` applies to newly generated outputs. Existing run directories retain their recorded artifacts.

The dataset SHA-256 fingerprint includes the codebook bytes, selected training-entry metadata, ranking bytes, and encoding bytes, plus the same information for all saved validation entries, with explicit boundaries between components.

Checkpoint saving is **disabled by default**. Both `CHECKPOINT_START_EPOCH` and `CHECKPOINT_END_EPOCH` near the top of `train.py` default to `None`. Enable saving by supplying `--checkpoint-start-epoch` and `--checkpoint-end-epoch` together, or by setting both module constants to integer epochs. Supplying only one bound raises an argument error.

The bounds are inclusive and must be positive, with the end epoch at least the start epoch. They can extend beyond the training budget. The actual checkpoint epochs are the intersection of the configured window with completed training epochs. The run and combination configurations record both supplied bounds, or two `null` values when saving is disabled. Default runs produce zero model checkpoint files at every training length.

A checkpoint containing the 566,327-parameter float32 model and Adam moment states is approximately **6.8 MB**. The explicit 30–40 range in section 1 saves 11 per combination and 176 overall, using roughly **1.2 GB** for checkpoints, plus reports and plots, using decimal MB and GB.

Each checkpoint contains `epoch`, `model_state_dict`, `optimizer_state_dict`, `config`, `channels`, `selected_instance_ids`, `fitting_instance_ids`, `validation_instance_ids`, `train_metrics`, and `validation_metrics`. The configuration and split memberships identify the experiment and data associated with that saved epoch. Architecture metadata identifies the normalization, channel counts, strides, and tensor shapes used by the saved model. Earlier two-convolution checkpoints and original four-convolution checkpoints require their corresponding model definitions for restoration. Check the full metadata together with the architecture label when choosing the matching definition.

To inspect a checkpoint you have generated, choose its combination and epoch explicitly. This example loads combination 1 at epoch 30 from the explicit 30–40 range command in section 1.

```python
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path("model-training").resolve()))
from model import PairwiseCNNVAE

checkpoint = torch.load(
    "model-training/runs/comparison_30_to_40/combination_01/checkpoints/epoch_030.pt",
    map_location="cpu",
    weights_only=True,
)
assert checkpoint["epoch"] == 30
assert checkpoint["config"]["architecture"]["architecture_version"] == "model_arch_1"
assert checkpoint["config"]["architecture"]["normalization"] == "LayerNorm"
assert checkpoint["config"]["architecture"]["layer_norm_axes"] == ["channel", "height", "width"]
assert checkpoint["config"]["architecture"]["layer_norm_eps"] == 1e-5
assert checkpoint["config"]["architecture"]["layer_norm_elementwise_affine"] is True
assert checkpoint["config"]["architecture"]["encoder_layer_norm_shapes"] == [
    [32, 8, 96], [64, 4, 48], [32, 2, 24]
]
assert checkpoint["config"]["architecture"]["decoder_layer_norm_shapes"] == [
    [64, 4, 48], [32, 8, 96]
]
assert checkpoint["config"]["architecture"]["stride"] == [2, 2]
assert checkpoint["config"]["architecture"]["encoder_channels"] == [22, 32, 64, 32]
assert checkpoint["config"]["architecture"]["decoder_channels"] == [32, 64, 32, 22]
assert checkpoint["config"]["architecture"]["encoded_shape"] == [32, 2, 24]
assert checkpoint["config"]["architecture"]["flattened_features"] == 1536
assert checkpoint["config"]["combination_id"] == "combination_01"
assert checkpoint["config"]["checkpoint_start_epoch"] == 30
assert checkpoint["config"]["checkpoint_end_epoch"] == 40
assert checkpoint["config"]["hyperparameters"] == {
    "learning_rate": 0.0001,
    "lambda_rec": 0.5,
    "lambda_kl": 0.001,
    "lambda_rank": 0.5,
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

The assertions verify the architecture version, normalization axes, channel counts, stride, encoded shape, flattened size, saved epoch, checkpoint window, combination ID, and hyperparameters against the chosen grid row. `map_location="cpu"` maps the saved model and optimizer tensors to CPU during loading.

## 10. Verification and existing artifacts

Run the behavioral checks from the repository root.

```bash
.venv/bin/python -m unittest discover -s model-training/tests -v
```

The two-epoch command in section 1 exercises all 16 independent experiments, per-epoch measurements, JSON/CSV export, and curves. The 21-epoch command additionally exercises an explicit single-epoch checkpoint range. Use a fresh output directory for each invocation.

Existing directories under `model-training/runs/` contain artifacts from earlier training workflows. Their saved configurations and reports identify the procedure and metric schema used for those runs. The experiment layout, commands, and checkpoint policy above describe newly generated runs under the current workflow.
