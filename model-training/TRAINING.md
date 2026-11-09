# CNN-VAE and pairwise fun-ranking training

This pipeline implements the joint training stage of the project proposal with architecture **`model_arch_2`**: a shared four-stage CNN-VAE with LayerNorm and ReLU, plus a fully connected classifier on two concatenated latent samples. The classifier predicts whether the first playthrough is preferred to the second. [Project proposal](player-exp-pairwise-cnn-vae-pcgml-proposal.pdf), [current architecture specification](model_arch_2.md).

The workflow reads its grid from [hyperparameter-combinations.txt](hyperparameter-combinations.txt), currently **20 independent combinations for 50 epochs each**. Every epoch records fitting and validation measurements, and each combination produces curves for losses, overall accuracy, and normalized tile Hamming distance. You choose a combination and epoch from those records. Each experiment starts with a fresh model and uses the same saved `train_set` and `validation_set` assignments. Supplying both checkpoint-range bounds enables model saving and an input/reconstruction preview for each saved model, using five randomly selected validation levels. The modules cover [data loading](data.py), [the model](model.py), [experiment execution](train.py), [curves](plotting.py), and [checkpoint previews](reconstruction_previews.py).

## 1. Setup and commands

Run these commands from the repository root. The training dependencies are pinned in [requirements.txt](requirements.txt). Reuse a root `.venv` if it exists:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r model-training/requirements.txt
```

The environment uses Python **3.14.7** with pinned PyTorch **2.15.0**, NumPy **2.3.4**, Matplotlib **3.11.2**, and Pillow **12.0.0**. Matplotlib generates each combination's `validation_curves.png`; Pillow supports the existing level renderer and checkpoint comparison sheets.

For a small end-to-end run, create a separate one-combination file and train with four original comparisons for two epochs. Saving both epochs also exercises validation reconstruction previews:

```bash
cat > /tmp/mario-smoke-hyperparameters.txt <<'EOF'
learning_rate: 1e-4
lambda_kl: 0.001
lambda_rec: 1
lambda_rank: 0.01
EOF

.venv/bin/python model-training/train.py \
  --hyperparameters-file /tmp/mario-smoke-hyperparameters.txt \
  --num-training-samples 4 \
  --epochs 2 \
  --checkpoint-start-epoch 1 \
  --checkpoint-end-epoch 2 \
  --output-dir model-training/runs/model_arch_2_smoke
```

Open `model-training/runs/model_arch_2_smoke/REPORT.md` after the run to see its curves and both checkpoint comparison sheets. The temporary file leaves the main hyperparameter file unchanged.

For all combinations in the main file, currently 20, using the complete supplied training portion and the 50-epoch default, use

```bash
.venv/bin/python model-training/train.py \
  --output-dir model-training/runs/comparison_50_epochs
```

This command records measurements and curves with checkpoint saving disabled.

An explicit single-epoch checkpoint range can be exercised with four comparisons and two training epochs per combination:

```bash
.venv/bin/python model-training/train.py \
  --num-training-samples 4 \
  --epochs 2 \
  --checkpoint-start-epoch 2 \
  --checkpoint-end-epoch 2 \
  --output-dir model-training/runs/comparison_checkpoint_check
```

That run saves one epoch-2 checkpoint and its reconstruction previews per combination.

To train for the default 50 epochs and save epochs 30 through 40 inclusive, use

```bash
.venv/bin/python model-training/train.py \
  --checkpoint-start-epoch 30 \
  --checkpoint-end-epoch 40 \
  --output-dir model-training/runs/comparison_30_to_40
```

This saves 11 checkpoints per combination, giving 220 across the current 20 combinations, each with reconstruction previews. The window controls saving while each experiment completes its full training budget. Changing the hyperparameter file changes the number of combinations and total saved models.

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

This rule keeps all pairs from the same ranking and all other linked observations together. The saved fitting/validation split uses the same grouping and is shared by all combinations. `run_config.json` records requested and actual comparison counts, `selected_train_comparisons.json` records the complete selected training entries, and `selected_validation_comparisons.json` records the saved validation entries.

## 4. Ordered inputs and targets

For every retained comparison `preferred > other`, training presents both views:

| Input A | Input B | Target y |
|---|---|---:|
| Preferred playthrough | Other playthrough | 1 |
| Other playthrough | Preferred playthrough | 0 |

Thus the complete `train_set` yields **464 ordered examples** with 232 targets of each class. The saved validation portion yields 116 ordered examples with 58 targets of each class. Reversal occurs after sample selection and after the fitting/validation split, so both orientations remain in the same group and partition. Fitting batches are shuffled each epoch, using a seeded DataLoader with `num_workers=0`.

Each original comparison has equal weight. There is no extra source-annotation normalization, duplicate removal, class weighting, or resampling. Consequently, a three-pair annotation contributes three original comparisons, exactly as stored in the prepared dataset. Reversing every pair doubles all comparison contributions equally. Repeated appearances of a playthrough remain part of their respective comparison losses.

## 5. Shared model architecture

The current architecture is **`model_arch_2`**, specified in [model_arch_2.md](model_arch_2.md) and implemented in [model.py](model.py). It retains four encoder convolutions and four decoder transposed convolutions. LayerNorm and ReLU follow every encoder convolution and the first three decoder convolutions. The FC-Tail uses ReLU and one dropout layer with probability **0.2** after its first hidden FC layer.

Both inputs pass independently through the same encoder parameters. Their sampled latent vectors use the same decoder parameters, and all ordered pairs use the same FC-Tail. All convolutions use **4×4 kernels**, stride **2**, and padding **1**. Each encoder convolution halves the spatial dimensions, and each transposed convolution doubles them.

| Encoder layer | Channels or FC dimensions | Output shape per playthrough | Following operations |
|---|---|---|---|
| Input | 22 channels | 22 × 16 × 192 | — |
| Conv 1 | 22 → 32, kernel 4×4, stride 2, padding 1 | 32 × 8 × 96 | LayerNorm((32, 8, 96)), ReLU |
| Conv 2 | 32 → 64, kernel 4×4, stride 2, padding 1 | 64 × 4 × 48 | LayerNorm((64, 4, 48)), ReLU |
| Conv 3 | 64 → 128, kernel 4×4, stride 2, padding 1 | 128 × 2 × 24 | LayerNorm((128, 2, 24)), ReLU |
| Conv 4 | 128 → 128, kernel 4×4, stride 2, padding 1 | 128 × 1 × 12 | LayerNorm((128, 1, 12)), ReLU |
| Flatten | 128 × 1 × 12 → 1,536 | 1,536 | — |
| Hidden FC | 1,536 → 128 | 128 | ReLU |
| Mean head | 128 → 32 | μ of shape 32 | Linear output |
| Log-variance head | 128 → 32 | log_var of shape 32 | Linear output |

LayerNorm normalizes over the complete `(channel, height, width)` feature map independently for each playthrough, with `eps=1e-5` and learned elementwise scale and bias. The encoder hidden FC uses ReLU, and the Gaussian heads return affine outputs. The diagonal Gaussian posterior is sampled by reparameterization:

```text
z = μ + exp(0.5 * log_var) * ε,   ε ~ N(0, I)
```

Each playthrough has a 32-dimensional latent sample. Sampling is used in optimization, reported measurements, and saved-model reconstruction previews.

| Decoder layer | Channels or FC dimensions | Output shape per playthrough | Following operations |
|---|---|---|---|
| Projection FC | 32 → 1,536 | 1,536 | Linear output |
| Reshape | 1,536 → 128 × 1 × 12 | 128 × 1 × 12 | — |
| Transposed conv 1 | 128 → 128, kernel 4×4, stride 2, padding 1 | 128 × 2 × 24 | LayerNorm((128, 2, 24)), ReLU |
| Transposed conv 2 | 128 → 64, kernel 4×4, stride 2, padding 1 | 64 × 4 × 48 | LayerNorm((64, 4, 48)), ReLU |
| Transposed conv 3 | 64 → 32, kernel 4×4, stride 2, padding 1 | 32 × 8 × 96 | LayerNorm((32, 8, 96)), ReLU |
| Transposed conv 4 | 32 → 22, kernel 4×4, stride 2, padding 1 | 22 × 16 × 192 | Linear logits |

The decoder projection remains linear before reshaping. Its first three transposed convolutions use LayerNorm over their complete per-playthrough feature maps, followed by ReLU. The final transposed convolution returns unrestricted logits for all 22 input bits. The reconstruction loss applies BCE-with-logits to these outputs.

The FC-Tail concatenates samples in the supplied input order:

```text
[z_A; z_B] of size 64
  → Linear(64,64) → ReLU → Dropout(0.2)
  → Linear(64,32) → ReLU
  → Linear(32,1)
```

The final scalar is an affine logit. Sigmoid gives `P(first playthrough preferred)` for probabilities and classification. Both tail hidden layers use ReLU; dropout follows only the first hidden layer. It is active during optimization and disabled during evaluation. There is no max pooling or unpooling.

### Parameter count

The implemented model has **1,312,151 parameters**, all trainable. Shared modules are counted once.

| Component | Trainable parameters |
|---|---:|
| Shared encoder, including mean and log-variance heads | 731,680 |
| Shared decoder | 574,198 |
| Pairwise FC-Tail | 6,273 |
| **Total** | **1,312,151** |

The layer counts include convolutional and fully connected weights and biases. LayerNorm contributes one learned scale and one learned shift per normalized feature-map element, giving `2 × C × H × W` parameters for each normalized shape.

| Component | Layer | Parameters |
|---|---|---:|
| Encoder | Conv 22 → 32, 4×4, followed by LayerNorm((32, 8, 96)) | 60,448 |
| Encoder | Conv 32 → 64, 4×4, followed by LayerNorm((64, 4, 48)) | 57,408 |
| Encoder | Conv 64 → 128, 4×4, followed by LayerNorm((128, 2, 24)) | 143,488 |
| Encoder | Conv 128 → 128, 4×4, followed by LayerNorm((128, 1, 12)) | 265,344 |
| Encoder | Hidden FC 1,536 → 128 | 196,736 |
| Encoder | Mean head 128 → 32 | 4,128 |
| Encoder | Log-variance head 128 → 32 | 4,128 |
| Decoder | Projection FC 32 → 1,536 | 50,688 |
| Decoder | Transposed conv 128 → 128, 4×4, followed by LayerNorm((128, 2, 24)) | 274,560 |
| Decoder | Transposed conv 128 → 64, 4×4, followed by LayerNorm((64, 4, 48)) | 155,712 |
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

The loss implementation uses `torch.nn.functional.binary_cross_entropy_with_logits` with its mean reduction. See [loss_components and weighted_loss in train.py](train.py).

## 7. Independent experiments and the fixed split

The experiment grid is read at startup from [hyperparameter-combinations.txt](hyperparameter-combinations.txt). Edit its comma-separated values to choose the training runs:

```text
learning_rate: 1e-4, 1e-5, 5e-5, 1e-6, 5e-6
lambda_kl: 0.001
lambda_rec: 1, 3
lambda_rank: 0.01, 1
```

These are the values currently saved in the file. The parser expands the Cartesian product, giving **5 × 1 × 2 × 2 = 20 combinations**. Ordering follows learning rate, KL weight, reconstruction weight, then ranking weight, preserving value order within each list. For the file above, `combination_01` uses `(learning_rate=1e-4, lambda_rec=1, lambda_kl=0.001, lambda_rank=0.01)`, and `combination_20` uses `(5e-6, 3, 0.001, 1)`.

Each output folder is named `learning_rate_lambda_rec_lambda_kl_lambda_rank`, in that order. Values between zero and one use scientific notation without zero-padded exponents; whole numbers omit `.0`. For example, `(1e-4, 1, 1e-3, 1)` produces `1e-4_1_1e-3_1/`. The first and last folders for the current file are `1e-4_1_1e-3_1e-2/` and `5e-6_3_1e-3_1/`. Number formatting preserves the configured float precision so distinct parameter values retain distinct folder names. The stable `combination_01` through `combination_20` values remain experiment identifiers in metadata, reports, and console messages; `experiments.json` records each identifier's actual `directory`.

Use a dot for decimals, such as `0.01`. Blank lines and `#` comments are allowed. All four parameter names are required exactly once. Learning rates must be finite and positive; loss weights must be finite and non-negative. Duplicate values, unknown parameters, missing parameters, and invalid numbers fail before loading the dataset or creating a run directory.

`--hyperparameters-file path/to/grid.txt` selects another file for one invocation. The default file is resolved beside `train.py`, independently of the working directory. The source file path and the complete expanded combinations are saved in `run_config.json`; each combination's selected values are also saved in `config.json` and `experiments.json`. Editing the text file changes subsequent invocations.

Each combination receives a freshly initialized model and Adam optimizer. Resetting the configured seed, 42 by default, before each experiment gives the same initial model weights and fitting-data order across combinations. Adam uses its PyTorch defaults apart from the specified learning rate. The combination keeps its own learning rate and loss weights throughout its epoch budget.

The saved `validation_set` is authoritative: fitting uses the selected `train_set` comparisons, and validation uses all saved validation comparisons across all experiments. Training does not split `train_set` a second time. The preprocessing default takes 20% of the original 290 training comparisons using whole groups and seed 42, giving **232 fitting comparisons and 58 validation comparisons**. Both input orders are then generated inside their assigned portion. `run_config.json` records `validation_source` as `validation_set` and sets its internal `validation_fraction` to `null` because no internal split was performed.

Each combination fits on those 232 comparisons for its entire **50-epoch** default budget. The fixed validation portion supports comparison of epochs and experiments. After all experiments finish, you choose the combination and epoch using the recorded measurements and curves. Every combination completes the requested epoch budget independently.

For legacy datasets without saved validation entries, training retains its internal split controlled by `--validation-fraction` (default `0.2`). It sorts and seeded-shuffles group IDs, then takes whole groups until reaching the target while retaining at least one fitting group. Indivisible groups can change the achieved fraction. If only one selected group is available in that legacy layout, validation measurements are recorded as `null`. In the saved layout, the training seed and sample cap do not change validation membership; change the preprocessing split to choose another validation set.

The main controls are available near the top of `train.py` and through the CLI.

| Control | Default | Meaning |
|---|---:|---|
| `DEFAULT_HYPERPARAMETERS` / `--hyperparameters-file` | `hyperparameter-combinations.txt` beside `train.py` | Parameter lists whose complete Cartesian product is trained |
| `NUM_TRAINING_SAMPLES` / `--num-training-samples` | `None` | Cap on original comparisons, preserving whole groups |
| `EPOCHS` / `--epochs` | 50 | Full epoch budget for every combination in the selected file |
| `CHECKPOINT_START_EPOCH` / `--checkpoint-start-epoch` | `None` | Optional first checkpoint epoch, inclusive, supplied together with the end epoch |
| `CHECKPOINT_END_EPOCH` / `--checkpoint-end-epoch` | `None` | Optional last checkpoint epoch, inclusive, supplied together with the start epoch |
| `--batch-size` | 16 | Ordered fitting examples per optimization batch |
| `--validation-fraction` | 0.2 | Fallback target fraction for legacy datasets without saved validation entries |
| `--seed` | 42 | Initialization, sample selection, split, and random-stream seed |
| `--device` | `auto` | CUDA when available, otherwise MPS when available, otherwise CPU |
| `--cpu-threads` | 4 | PyTorch CPU thread count |
| `--output-dir` | New timestamped directory | Root directory for all experiments in this invocation |

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

Python, NumPy, PyTorch, accelerator RNGs, and fitting-data shuffling are seeded. cuDNN benchmarking is disabled and its deterministic option is enabled when available. Numerical results can vary across devices, PyTorch releases, or batch sizes. The run records its environment and reporting convention.

## 9. Saved outputs, curves, and checkpoints

Each invocation creates one experiment root with the shared configuration and split.

| File | Contents |
|---|---|
| `run_config.json` | Dataset and fingerprint, requested/actual sample counts, source hyperparameter path and expanded combinations, epoch budget, checkpoint bounds, seeds, selected preview inputs, device, environment, architecture, and shared run settings |
| `channels.json` | Exact input channel codebook |
| `selected_train_comparisons.json` | Selected manifest entries with participant, playthrough, and grouping provenance |
| `selected_validation_comparisons.json` | Saved validation manifest entries when `validation_set` is present |
| `split.json` | Fixed fitting and validation memberships shared by all combinations |
| `experiments.json` | All combinations, their `pending`, `running`, `completed`, or `failed` status, and completed summaries with curve and checkpoint-preview paths |
| `REPORT.md` | Human-readable run settings, links to each combination's six-panel fitting/validation curves, and an embedded input/reconstruction sheet for each saved checkpoint after all combinations finish |

Each parameter-named directory, such as `1e-4_1_1e-3_1e-2/`, contains its own outputs. Its folder name records the learning rate, reconstruction weight, KL weight, and ranking weight; the number of directories follows the configured grid.

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
| `checkpoint_previews.json` | Index of saved checkpoint epochs and their sample counts, comparison PNGs, and sample manifests; updated after each checkpoint's preview is saved |
| `reconstructions/epoch_030/`, etc. | Validation inputs and VAE reconstructions associated with that saved checkpoint, described below |

The JSON history and complete epoch CSV are each replaced atomically after every epoch's measurements. This makes completed epoch measurements available while the experiment continues. Prediction CSV files summarize the final epoch. The history retains measurements for earlier epochs.

The overall-accuracy and normalized-Hamming-distance panels use a vertical range of 0–1. Higher accuracy indicates more correct preference predictions. Lower normalized Hamming distance indicates closer tile reconstruction. The filename `validation_curves.png` applies to newly generated outputs. Existing run directories retain their recorded artifacts.

The dataset SHA-256 fingerprint includes the codebook bytes, selected training-entry metadata, ranking bytes, and encoding bytes, plus the same information for all saved validation entries, with explicit boundaries between components.

Checkpoint saving is **disabled by default**. Both `CHECKPOINT_START_EPOCH` and `CHECKPOINT_END_EPOCH` near the top of `train.py` default to `None`. Enable saving by supplying `--checkpoint-start-epoch` and `--checkpoint-end-epoch` together, or by setting both module constants to integer epochs. Supplying only one bound raises an argument error.

The bounds are inclusive and must be positive, with the end epoch at least the start epoch. They can extend beyond the training budget. The actual checkpoint epochs are the intersection of the configured window with completed training epochs. The run and combination configurations record both supplied bounds, or two `null` values when saving is disabled. Default runs produce zero model checkpoint files at every training length.

A checkpoint containing the 1,312,151-parameter float32 model and Adam moment states is approximately **15.8 MB**; the two-epoch smoke check produced 15,806,509-byte checkpoint files. The explicit 30–40 range in section 1 saves 11 per combination and 220 overall for the current grid, using roughly **3.5 GB** for checkpoints, plus reconstruction images, arrays, reports, and plots. These are decimal MB and GB; serialized size can vary with configuration metadata and library versions.

Each checkpoint contains `epoch`, `model_state_dict`, `optimizer_state_dict`, `config`, `channels`, `selected_instance_ids`, `fitting_instance_ids`, `validation_instance_ids`, `train_metrics`, and `validation_metrics`. The configuration and split memberships identify the experiment and data associated with that saved epoch. Architecture metadata identifies `model_arch_2`, LayerNorm shapes, channel counts, strides, activations, and dropout. Earlier architecture checkpoints require their corresponding model definitions for restoration. Check the full metadata together with the architecture label when choosing the matching definition.

### Validation reconstruction previews for every saved model

At startup, the pipeline randomly selects **five distinct validation level IDs** using the configured `--seed` (42 by default). It sorts validation entries by instance ID and their playthroughs by level ID and playthrough key, retains the first occurrence of each level ID, then samples without replacement from the sorted level IDs. This deduplication applies only to visualization; training and validation measurements retain their original playthrough observations. If fewer than five level IDs are available, all available levels are selected without repetition. Saved `validation_set` entries are used when present; otherwise the same policy uses the pipeline's internal validation portion.

The same selected inputs and their recorded paths are reused across every saved epoch and hyperparameter combination. Immediately after saving a checkpoint, its current model encodes those inputs, draws one posterior latent sample per input, and decodes the samples. These images are **reconstructions of validation inputs**, rather than unconditional samples from the standard normal prior.

Preview inference uses evaluation mode, disabled gradients, and the seed `(seed + 2) % (2**32)`, which is 44 by default. Reusing the seed gives the same latent-noise sequence for comparison as model parameters change. The previous model mode and the training RNG states are restored after preview generation, so previews do not change subsequent optimization.

Decoder logits become a valid binary `(22, 16, 192)` encoding before rendering. At every cell, argmax over channels 0–20 selects exactly one tile. Channel 21 is decoded independently with `logit >= 0`, equivalent to path probability at least 0.5. Both the original and reconstructed arrays are rendered by `encoding_to_image` from the existing [encoding_to_level_image.py](../datasets/fun-study-dataset-analysis/encoding_to_level_image.py). Black X markers are drawn over the tiles to show the corresponding path.

For example, a saved epoch-30 model has these preview files beneath its combination directory:

```text
checkpoint_previews.json
reconstructions/epoch_030/
    01_input.png
    01_generated.png
    01_generated.npy
    ... through 05 for five selected levels ...
    channels.json
    samples.json
    comparison.png
```

Each individual PNG preserves the complete 192-column level. `comparison.png` stacks labeled input/reconstruction pairs for all selected levels. `samples.json` records the checkpoint, epoch, input-selection and posterior-sampling seeds, decoding policy, level IDs, playthrough keys, source instance IDs and encoding paths, and output filenames. `channels.json` supplies the exact codebook used by the renderer. The source encoding paths are relative to the dataset root; the checkpoint path in this manifest is relative to its combination directory.

The console prints each completed comparison-sheet path. `checkpoint_previews.json` makes the available sheets discoverable while training continues, and the run's final `REPORT.md` embeds a sheet for every saved model. Preview directories are created only for epochs whose checkpoints are saved. If validation is unavailable, each saved epoch gets a `samples.json` explaining the skip and no PNGs. When checkpoint saving is disabled, no checkpoint previews are produced.

To inspect a checkpoint you have generated, choose its parameter folder and epoch explicitly. This example loads logical experiment `combination_01` from `1e-4_1_1e-3_1e-2/` at epoch 30 from the explicit 30–40 range command in section 1.

```python
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path("model-training").resolve()))
from model import PairwiseCNNVAE

checkpoint = torch.load(
    "model-training/runs/comparison_30_to_40/1e-4_1_1e-3_1e-2/checkpoints/epoch_030.pt",
    map_location="cpu",
    weights_only=True,
)
assert checkpoint["epoch"] == 30
assert checkpoint["config"]["architecture"]["architecture_version"] == "model_arch_2"
assert checkpoint["config"]["architecture"]["normalization"] == "LayerNorm"
assert checkpoint["config"]["architecture"]["layer_norm_axes"] == ["channel", "height", "width"]
assert checkpoint["config"]["architecture"]["layer_norm_eps"] == 1e-5
assert checkpoint["config"]["architecture"]["layer_norm_elementwise_affine"] is True
assert checkpoint["config"]["architecture"]["encoder_layer_norm_shapes"] == [
    [32, 8, 96], [64, 4, 48], [128, 2, 24], [128, 1, 12]
]
assert checkpoint["config"]["architecture"]["decoder_layer_norm_shapes"] == [
    [128, 2, 24], [64, 4, 48], [32, 8, 96]
]
assert checkpoint["config"]["architecture"]["stride"] == [2, 2]
assert checkpoint["config"]["architecture"]["encoder_channels"] == [22, 32, 64, 128, 128]
assert checkpoint["config"]["architecture"]["decoder_channels"] == [128, 128, 64, 32, 22]
assert checkpoint["config"]["architecture"]["encoded_shape"] == [128, 1, 12]
assert checkpoint["config"]["architecture"]["flattened_features"] == 1536
assert checkpoint["config"]["architecture"]["encoder_convolution_activation"] == "ReLU"
assert checkpoint["config"]["architecture"]["decoder_projection_activation"] == "linear"
assert checkpoint["config"]["architecture"]["dropout"] == 0.2
assert checkpoint["config"]["architecture"]["dropout_after_fc_layers"] == [1]
assert checkpoint["config"]["combination_id"] == "combination_01"
assert checkpoint["config"]["checkpoint_start_epoch"] == 30
assert checkpoint["config"]["checkpoint_end_epoch"] == 40
assert checkpoint["config"]["hyperparameters"] == {
    "learning_rate": 0.0001,
    "lambda_rec": 1.0,
    "lambda_kl": 0.001,
    "lambda_rank": 0.01,
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

The assertions verify the architecture version, normalization axes, channel counts, stride, encoded shape, flattened size, activations, dropout, saved epoch, checkpoint window, combination ID, and hyperparameters against the current first grid row. If you edit the hyperparameter file, adjust the example's expected values to match that run's saved configuration. `map_location="cpu"` maps the saved model and optimizer tensors to CPU during loading.

## 10. Verification and existing artifacts

Run the behavioral checks from the repository root.

```bash
.venv/bin/python -m unittest discover -s model-training/tests -v

PYTHONPATH=datasets/fun-study-dataset-analysis \
  .venv/bin/python -m unittest discover \
  -s datasets/fun-study-dataset-analysis/tests \
  -p test_encoding_to_level_image.py -v
```

Verification for `model_arch_2` passed **23 training/preview tests and 7 existing renderer tests**. A real-data smoke run trained one combination for two epochs with one original fitting comparison, evaluated all 58 saved validation comparisons, and saved two checkpoints. Both checkpoints reloaded strictly with the `model_arch_2` metadata and produced previews for the same five validation levels. The integration tests also compare training histories with and without preview generation to verify that rendering preserves the training random streams. This check does not constitute a complete 20-combination, 50-epoch training run.

The two-epoch command in section 1 provides a reproducible one-combination check of per-epoch measurements, JSON/CSV export, curves, checkpoints, and reconstruction previews. Use a fresh output directory for each invocation.

Existing directories under `model-training/runs/` contain artifacts from earlier training workflows. Their saved configurations and reports identify the procedure and metric schema used for those runs. The experiment layout, commands, and checkpoint policy above describe newly generated runs under the current workflow.
