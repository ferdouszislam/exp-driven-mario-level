# CNN-VAE + Pairwise FC-Tail Architecture Specification

This document specifies the current `initial_model_architecture_crop16` network. It retains the original **shared CNN-VAE** and **concatenated-latent FC-Tail** approach, with dimensions updated for the 176-column export after preprocessing removes the first 16 source columns from both tiles and paths.

## 1. CNN Encoder

**Input:** `(batch_size, 22, 16, 176)` in channels-first (NCHW) format.

Each level has **16 rows × 176 columns** after tiles and paths are cropped to source columns `16:192`. This gives **2,816 grid cells** and **61,952 binary channel values** per playthrough.

| Layer | In → Out channels | Kernel | Stride | Padding | Output shape (C × H × W) |
|---|---|---|---|---|---|
| Conv2D 1 | 22 → 32 | 4×4 | 2 | 1 | 32 × 8 × 88 |
| Conv2D 2 | 32 → 64 | 4×4 | 2 | 1 | 64 × 4 × 44 |
| Conv2D 3 | 64 → 128 | 4×4 | 2 | 1 | 128 × 2 × 22 |
| Conv2D 4 | 128 → 128 | 4×4 | 2 | 1 | 128 × 1 × 11 |
| Flatten | — | — | — | — | 1,408 |
| FC | 1,408 → 128 | — | — | — | 128 |
| FC (μ) | 128 → 32 | — | — | — | 32 |
| FC (log σ²) | 128 → 32 | — | — | — | 32 |

- **Activation:** LeakyReLU with negative slope `0.2` after each convolution and after the 128-unit fully connected layer.
- **Normalization:** GroupNorm after each convolution, before LeakyReLU. The implementation uses 8 groups, a valid divisor of every normalized channel count.
- **Max pooling:** None. The strided convolutions perform downsampling.
- **Latent dimension:** `32`.
- **VAE sampling (reparameterization):** `z = μ + exp(0.5 * log_var) * ε`, with `ε ~ N(0, I)`.
- Each input playthrough is encoded independently using **the same encoder weights**.

## 2. CNN Decoder

**Input:** A 32-dimensional latent vector (`batch_size × 32`).

| Layer | In → Out channels | Kernel | Stride | Padding | Output shape (C × H × W) |
|---|---|---|---|---|---|
| FC | 32 → 1,408 | — | — | — | 1,408 |
| Reshape | — | — | — | — | 128 × 1 × 11 |
| ConvTranspose2D 1 | 128 → 128 | 4×4 | 2 | 1 | 128 × 2 × 22 |
| ConvTranspose2D 2 | 128 → 64 | 4×4 | 2 | 1 | 64 × 4 × 44 |
| ConvTranspose2D 3 | 64 → 32 | 4×4 | 2 | 1 | 32 × 8 × 88 |
| ConvTranspose2D 4 | 32 → 22 | 4×4 | 2 | 1 | 22 × 16 × 176 |

- **Activation:** LeakyReLU with negative slope `0.2` after the first three transposed convolutional layers.
- **Normalization:** GroupNorm after the first three transposed convolutions, before LeakyReLU.
- **Final output:** 22-channel **logits** (no final sigmoid in the model when using `BCEWithLogitsLoss`).
- **Reconstruction loss:** Binary cross-entropy across all 22 channels and all 16×176 grid positions, consistent with the proposal.
- **Max pooling/unpooling:** None.
- The decoder is shared between both playthroughs.

## 3. FC-Tail: Pairwise Preference Predictor

**Input:** Concatenation of two 32-dimensional sampled latent vectors in the given input order.

`[z_A; z_B]` has shape `(batch_size, 64)`.

| Layer | Input size | Output size | Activation |
|---|---:|---:|---|
| FC 1 | 64 | 64 | LeakyReLU(0.2) |
| Dropout | 64 | 64 | Dropout(0.1) |
| FC 2 | 64 | 32 | LeakyReLU(0.2) |
| FC 3 | 32 | 1 | Sigmoid |

- **Output:** `p(z_A, z_B)`, the probability that the **first** playthrough is preferred to the second.
- **Preference loss:** Binary cross-entropy against the pairwise target.
- If using `BCEWithLogitsLoss`, have the model return the FC 3 **logit** and apply sigmoid only when obtaining probabilities. This is numerically equivalent to a sigmoid output followed by BCE.
- Use the **same FC-Tail** for all ordered pairs.

## 4. Overall Architecture

```text
Input A: (22, 16, 176)              Input B: (22, 16, 176)
          |                                    |
          v                                    v
     SHARED CNN ENCODER                  SHARED CNN ENCODER
     Conv 22->32                         Conv 22->32
     Conv 32->64                         Conv 32->64
     Conv 64->128                        Conv 64->128
     Conv 128->128                       Conv 128->128
     Flatten -> FC 128                   Flatten -> FC 128
     μ: 32, log_var: 32                 μ: 32, log_var: 32
          |                                    |
          v                                    v
        z_A: 32                              z_B: 32
          |                                    |
          +-----------+------------+-----------+
          |           |            |           |
          v           |            |           v
     SHARED CNN       |            |      SHARED CNN
       DECODER        |            |        DECODER
     FC 1408          |            |      FC 1408
     Reshape          |            |      Reshape
     128 x 1 x 11     |            |      128 x 1 x 11
     Deconv 128       |            |      Deconv 128
     Deconv 64        |            |      Deconv 64
     Deconv 32        |            |      Deconv 32
     Deconv 22        |            |      Deconv 22
          |           |            |           |
          v           +-----+------+           v
  Reconstructed A           |          Reconstructed B
   (22, 16, 176)            v           (22, 16, 176)
                      Concatenate
                       [z_A; z_B]
                          64
                           |
                           v
                        FC-TAIL
                       64 -> 64
                       64 -> 32
                       32 -> 1
                       Sigmoid
                           |
                           v
                      P(A preferred)
```

### Architecture summary

- **Encoder:** 4 convolutional layers, 1 shared hidden FC layer, and 2 Gaussian parameter heads.
- **Decoder:** 1 projection FC layer, reshape, and 4 transposed convolutional layers.
- **Latent space:** 32 dimensions per playthrough.
- **FC-Tail:** 3 fully connected layers (64 → 64 → 32 → 1), with dropout after the first.
- **Pooling:** No max-pooling or unpooling.
- **Pair processing:** Two independent passes through the **same** encoder; their latent samples are concatenated in order for pairwise classification.
- **Reconstruction:** Each latent sample is decoded by the **same** decoder.
- **Joint training loss from the proposal:** `L = λ_rec * L_rec + λ_KL * L_KL + λ_rank * BCE(p(z_A, z_B), y)`.

**Status:** These layer dimensions are proposed starting hyperparameters, not exact configurations claimed to be taken from a cited paper.

The encoder widths are `176 → 88 → 44 → 22 → 11`; the decoder reverses those widths to produce exactly 176 columns without interpolation; all convolutions use the padding value 1 shown above. New checkpoints record `input_shape=[22,16,176]`, `encoded_shape=[128,1,11]`, `flattened_features=1408`, `source_width=192`, and `dropped_leading_columns=16`. The model has **1,117,591 trainable parameters**. Earlier 192-column checkpoints require their original model definition because their fully connected layers use 1,536 features.
