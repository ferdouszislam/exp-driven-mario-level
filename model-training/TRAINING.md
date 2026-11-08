# CNN-VAE and pairwise fun-ranking training

This pipeline implements the joint training stage of the [project proposal](player-exp-pairwise-cnn-vae-pcgml-proposal.pdf), using the exact layer dimensions in [initial_model_architecture.md](initial_model_architecture.md). It trains one shared CNN-VAE and a fully connected classifier on two concatenated latent samples. The classifier predicts whether the first playthrough is preferred to the second; it does not assign absolute numerical fun ratings.

The implementation consists of [data.py](data.py) for loading and ordered comparisons, [model.py](model.py) for the network, and [train.py](train.py) for hyperparameter selection, final training, saving, and train-set reporting. The current selection rule chooses the smallest absolute difference between fitting and validation **weighted total loss**, as requested. This stage does not implement test-set evaluation or latent optimization for level generation. Earlier saved runs used the previous ranking-BCE selector and are identified separately in section 11.

## 1. Setup and commands

Run these commands from the repository root. The training dependencies are pinned in [requirements.txt](requirements.txt). Reuse a root `.venv` if it exists:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r model-training/requirements.txt
```

The installed environment was checked with Python **3.14.7**, PyTorch **2.15.0**, and NumPy **2.3.4**. The training code does not require Pillow, image resizing, scikit-learn, or a PDF reader.

For a small end-to-end run with four original comparisons:

```bash
.venv/bin/python model-training/train.py \
  --num-training-samples 4 \
  --epochs 2 \
  --tuning-epochs 1 \
  --output-dir model-training/runs/smoke_total_loss_gap
```

For the full supplied training portion with the defaults:

```bash
.venv/bin/python model-training/train.py \
  --output-dir model-training/runs/full_train_total_loss_gap
```

Each output directory must be new or empty. Omit `--output-dir` to create a new UTC-timestamped directory under `model-training/runs/`. The script rejects output inside the input dataset. View all flags with:

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

This rule keeps all pairs from the same ranking and all other linked observations together. The same grouping also controls the temporary validation split. `config.json` records requested and actual comparison counts, and `selected_train_comparisons.json` records the complete selected entries.

## 4. Ordered inputs and targets

For every retained comparison `preferred > other`, training presents both views:

| Input A | Input B | Target y |
|---|---|---:|
| Preferred playthrough | Other playthrough | 1 |
| Other playthrough | Preferred playthrough | 0 |

Thus the complete train portion yields **580 ordered examples** with 290 targets of each class. Reversal occurs after sample selection and after the fitting/validation split, so both orientations remain in the same group and partition. Batches are shuffled each epoch, using a seeded DataLoader with `num_workers=0`.

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

## 7. Hyperparameter selection and final fitting

The proposal requires selecting the three loss weights and Adam learning rate on a grouped validation subset. The pipeline performs this selection **inside the selected training portion**. It never uses the supplied test split for selection.

By default, it targets 20% of selected original comparisons for temporary validation (`--validation-fraction 0.2`). Group IDs are sorted and seeded-shuffled. Whole groups are assigned until the target count is reached, while leaving at least one whole group for fitting. With all 290 supplied comparisons and seed 42, this gives **232 fitting comparisons and 58 validation comparisons**. The achieved fraction can differ for another subset or seed because groups cannot be divided. Source annotation IDs, related pairs, and reused playthroughs therefore stay together. The manifest's existing group definitions are used; no new participant-held-out or map-held-out split is claimed.

The declared small search uses these four joint candidates:

| Candidate | Adam learning rate | λ_rec | λ_KL | λ_rank |
|---|---:|---:|---:|---:|
| 1 | 0.001 | 1 | 0.001 | 1 |
| 2 | 0.001 | 1 | 0.01 | 1 |
| 3 | 0.0003 | 2 | 0.001 | 1 |
| 4 | 0.0003 | 1 | 0.001 | 2 |

These are explicit starting search choices; the proposal does not supply numerical values. Every candidate starts from the same seeded initialization and runs for **5 tuning epochs** by default. Adam uses its PyTorch defaults apart from the candidate learning rate.

After each candidate finishes fitting, the pipeline measures its fitting and validation losses with that **same fixed candidate model**. Both measurements use evaluation mode, disabled gradients, dropout off, and sampled latents with reporting seed `seed + 1`. The model is not updated during either measurement. Both portions use the same loss reductions and that candidate's existing weights:

```text
fitting_total_loss = λ_rec × fitting_reconstruction_BCE
                   + λ_KL × fitting_KL + λ_rank × fitting_ranking_BCE

validation_total_loss = λ_rec × validation_reconstruction_BCE
                      + λ_KL × validation_KL + λ_rank × validation_ranking_BCE

signed_gap = validation_total_loss - fitting_total_loss
selection_value = abs(signed_gap)
```

The selector chooses the candidate with the **smallest `selection_value`**. An exact tie is resolved only by the original candidate order. The fitting loss here is a fresh measurement of the final tuning model, rather than the last optimization epoch's averaged stochastic loss. No extra common validation weights, loss penalties, acceptance thresholds, or early stopping are introduced. Ranking BCE, accuracy, and binary F1-score remain reported measurements and do not select the candidate.

For example, suppose candidate A has fitting loss 0.80 and validation loss 0.83: its signed gap is `+0.03`, and its absolute gap is `0.03`. Candidate B has fitting loss 0.90 and validation loss 0.88: its signed gap is `-0.02`, and its absolute gap is `0.02`. The rule selects B because its absolute gap is smaller, even though its validation loss is higher. These are illustrative values, not results from a saved run. Taking the absolute value treats validation losses above and below fitting loss consistently.

The intent is to reduce fitting/validation mismatch associated with overfitting. A small gap does not establish good fit: both losses can be equally high, indicating underfitting. Candidate-specific weights also rescale the total-loss gap, and component differences can cancel in the weighted sum. Finally, a small gap after 5 tuning epochs does not guarantee avoidance of overfitting after the later 50-epoch fit. The implementation applies the requested gap rule directly and records the underlying losses so these limits remain visible.

For every candidate, `tuning.json` saves `fitting_metrics`, `validation_metrics`, `validation_minus_fitting_total_loss`, and `absolute_total_loss_gap`. Its top-level `selected_absolute_total_loss_gap` records the selected gap.

After selection, the tuning model is discarded. A freshly initialized model is trained on **all selected training comparisons**, including the temporary validation portion, for **50 final epochs** by default. Final epoch count is fixed; there is no early stopping or test-driven model selection.

If a very small run selects only one group, a grouped fitting/validation split is impossible. The script explicitly records `skipped_single_group_smoke_run`, uses candidate 1 provisionally, and still exercises final training and artifact creation. Such a run verifies the pipeline; it does not claim validation-based tuning.

Default controls are `batch_size=16` ordered examples, `seed=42`, and `cpu_threads=4`. `--device auto` chooses CUDA if available, otherwise Apple MPS if available, otherwise CPU. Explicit `--device cpu`, `--device cuda`, or `--device mps` can be supplied; an unavailable requested accelerator fails clearly.

## 8. Final train-only measurements

After final fitting, the script runs one reporting pass on exactly the selected original comparisons. It calls `model.eval()` and disables gradients. Dropout is off. With reporting seed **`seed + 2`**, it draws one posterior sample for each input occurrence and reuses those same two latent samples when reversing the order. A playthrough appearing in several comparison folders is sampled separately for each occurrence. The report does not replace sampled latents with posterior means or average over multiple draws.

For each pair, reconstruction BCE and KL are calculated once and averaged over its two inputs. Ranking BCE averages the forward target-1 and reverse target-0 losses. Dataset losses then average over original comparisons. These reductions correspond to the balanced ordered representation while avoiding a second random latent draw for the reverse direction.

Classification uses `sigmoid(logit) >= 0.5` to predict class 1 and assigns class 0 below that threshold. Accuracy and binary F1-score include **both input orders**. The positive class is **1**, meaning the first playthrough is preferred. The confusion matrix uses true-label rows and predicted-label columns in the order `[0, 1]`.

```text
accuracy = number of correct ordered predictions / number of ordered examples
TP = confusion_matrix[1][1]
FP = confusion_matrix[0][1]
FN = confusion_matrix[1][0]
f1_score = 2 × TP / (2 × TP + FP + FN)
```

The F1-score is defined as 0 when its denominator is zero. Saved metrics use `f1_score` for this positive-class result and `f1_positive_label=1` to identify the class. Each class's F1 remains available in `f1_by_class`, alongside `class_support` and `confusion_matrix`. Class 0 means the second playthrough is preferred.

Final reporting losses can differ from the last optimization epoch because the final model is fixed, dropout is off, and reporting uses its own posterior draws. The reported train accuracy and binary F1-score measure fit to the selected training data. They do not estimate performance on held-out comparisons.

Python, NumPy, PyTorch, available accelerator RNGs, and DataLoader shuffling are seeded. cuDNN benchmarking is disabled and its deterministic option is enabled when available. Exact numerical agreement across devices, PyTorch releases, or different batch sizes is not guaranteed; see PyTorch's [reproducibility guidance](https://docs.pytorch.org/docs/2.15/notes/randomness.html). The run records its environment and reporting seed.

## 9. Saved artifacts and reloading

Every completed run writes:

| File | Contents |
|---|---|
| `model.pt` | Final model state, Adam state, configuration/architecture, full channel codebook, selected instance IDs, and final train metrics |
| `config.json` | Dataset location and fingerprint, requested/actual comparison counts, epochs, device, seed, selected hyperparameters, selection rule and selected tuning gap, architecture, environment, parameter count, and elapsed time |
| `selected_train_comparisons.json` | Selected training manifest entries with participant/playthrough and grouping provenance |
| `tuning.json` | Fitting/validation instance IDs, candidate settings and histories, post-fit `fitting_metrics` and `validation_metrics`, signed `validation_minus_fitting_total_loss`, `absolute_total_loss_gap`, and the selected `selected_absolute_total_loss_gap`; or an explicit single-group smoke-run fallback |
| `train_history.json` | Per-epoch final-fitting losses and epoch duration |
| `train_metrics.json` | Final loss components, weighted total, accuracy, binary `f1_score` with `f1_positive_label=1`, per-class F1 diagnostics, confusion matrix, class supports, and reporting convention |
| `train_predictions.csv` | Every ordered train prediction, its target, probability, and comparison/level IDs |
| `REPORT.md` | Human-readable final train results, run settings, selection rule, and the selected tuning candidate's fitting/validation total losses and gap |

The dataset SHA-256 fingerprint includes the channel-codebook bytes and selected training-entry metadata, ranking bytes, and encoding bytes, with explicit boundaries between components. No test-entry metadata or test-file contents enter this fingerprint.

Model-state tensors are saved on CPU. The optimizer state is also retained and may originate from the training device. To reload the model for inspection on CPU, run from the repository root:

```python
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path("model-training").resolve()))
from model import PairwiseCNNVAE

checkpoint = torch.load(
    "model-training/runs/full_train/model.pt",
    map_location="cpu",
    weights_only=True,
)
model = PairwiseCNNVAE()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(checkpoint["train_metrics"])
```

`map_location="cpu"` also remaps saved optimizer tensors during loading. The example loads the historical `full_train` checkpoint described in section 11; use a new run's checkpoint path to inspect a model selected by the current total-loss-gap rule. This example restores the trained network; the CLI does not provide resume-training support.

## 10. Verification

Run the pipeline checks from the repository root:

```bash
.venv/bin/python -m unittest discover -s model-training/tests -v
```

The small training command in section 1 exercises loading, grouped selection, hyperparameter selection when at least two groups are available, forward/backward passes, final reporting, and checkpoint creation. Inspect its `REPORT.md`, `train_metrics.json`, and `tuning.json` to distinguish a smoke run from a full-data training run. Use a fresh output directory when repeating it.

Numerical results belong to each run's saved report. The implementation does not infer fun ratings from filenames, train on rendered PNGs, or consult the test tensors when producing those results.

## 11. Historical runs from the previous selector

The existing `full_train` and `smoke` directories were produced **before the total-loss-gap selector was introduced**. They used the previous rule: lowest validation ranking BCE, with reconstruction BCE as a tie-breaker. Their saved models, selection records, and measurements are historical results of that earlier rule; they have not been relabeled as results of the current selector.

Both runs also predate binary-F1 reporting. Their saved `macro_f1` values record the mean of the two class F1 values, and their original reports and checkpoints retain that metric. The `f1_score` schema described in sections 8–9 applies to newly generated measurements.

The historical [full training report](runs/full_train/REPORT.md) records the completed 50-epoch CPU run on all 290 comparisons. Hyperparameter selection used 232 comparisons for fitting and 58 for temporary validation, with no shared annotation groups or recorded playthroughs across that temporary split. Candidate 2 was selected by the previous ranking-BCE rule, then final fitting started from scratch on all 290 comparisons.

The [saved model](runs/full_train/model.pt), [exact metrics](runs/full_train/train_metrics.json), and [580 ordered predictions](runs/full_train/train_predictions.csv) are available in that run directory. Reloading the checkpoint on CPU reproduced every reported loss, metric, and exported prediction exactly. Independently counting the CSV predictions gave 575 correct predictions out of 580. All nine behavioral tests passed.

The separate historical [smoke report](runs/smoke/REPORT.md) records the four-comparison, two-epoch pipeline check under the previous selector. Its checkpoint was also reloaded and its metrics reproduced. Both historical output directories remain available. The commands in section 1 use new output directory names for the current selector; choose another new directory or omit `--output-dir` when rerunning them.

## 12. Verification of the current selector

The [total-loss-gap smoke report](runs/smoke_total_loss_gap/REPORT.md) records a completed run with four original comparisons, one tuning epoch per candidate, and two final epochs. Its [tuning record](runs/smoke_total_loss_gap/tuning.json) includes both post-fit total losses and both gap values for all four candidates. Candidate 2 had the smallest absolute gap, approximately `0.00507331`, and was selected by the new rule. This is a pipeline check, rather than a full-data performance result.

This smoke run also predates binary-F1 reporting. Its saved `macro_f1` and original report retain the recorded macro-F1 measurement. Its total-loss-gap selection record remains a verification of that selector.

The saved losses and selected gap were independently checked against the formula above. The [smoke checkpoint](runs/smoke_total_loss_gap/model.pt) records the new selection rule and gap, and reloading it reproduced the saved final train metrics exactly. A separate single-group CLI check confirmed that the gap is explicitly unavailable (`null`) when grouped validation cannot be performed. At that stage, all **12 behavioral tests** passed, including tests that distinguish absolute-gap selection from lowest-loss selection, check post-fit measurements and exact ties, and exercise the single-group fallback.

The full 290-comparison model has **not been retrained with this new selector**. Its existing checkpoint remains the historical result described in section 11. `smoke_total_loss_gap` now exists too, so choose a new output directory or omit `--output-dir` for another run.
