# CNN-VAE + Pairwise FC-Tail Architecture Specification

This document specifies the network architecture for implementing the proposed Mario level generation model. It retains the original **shared CNN-VAE** and **concatenated-latent FC-Tail** approach.

## 1. CNN Encoder

**Input:** `(batch_size, 22, 16, 192)` in channels-first (NCHW) format.

| Layer | In → Out channels | Kernel | Stride | Padding | Output shape (C × H × W) |
|---|---|---|---|---|---|
| Conv2D 1 | 22 → 32 | 4×4 | 2 | 1 | 32 × 8 × 96 |
| Conv2D 2 | 32 → 64 | 4×4 | 2 | 1 | 64 × 4 × 48 |
| Conv2D 3 | 64 → 128 | 4×4 | 2 | 1 | 128 × 2 × 24 |
| Conv2D 4 | 128 → 128 | 4×4 | 2 | 1 | 128 × 1 × 12 |
| Flatten | — | — | — | — | 1,536 |
| FC | 1,536 → 128 | — | — | — | 128 |
| FC (μ) | 128 → 32 | — | — | — | 32 |
| FC (log σ²) | 128 → 32 | — | — | — | 32 |

- **Activation:** LeakyReLU with negative slope `0.2` after each convolution and after the 128-unit fully connected layer.
- **Normalization:** GroupNorm after each convolution, before LeakyReLU. The number of groups is not specified; select a valid value dividing each channel count.
- **Max pooling:** None. The strided convolutions perform downsampling.
- **Latent dimension:** `32`.
- **VAE sampling (reparameterization):** `z = μ + exp(0.5 * log_var) * ε`, with `ε ~ N(0, I)`.
- Each input playthrough is encoded independently using **the same encoder weights**.

## 2. CNN Decoder

**Input:** A 32-dimensional latent vector (`batch_size × 32`).

| Layer | In → Out channels | Kernel | Stride | Padding | Output shape (C × H × W) |
|---|---|---|---|---|---|
| FC | 32 → 1,536 | — | — | — | 1,536 |
| Reshape | — | — | — | — | 128 × 1 × 12 |
| ConvTranspose2D 1 | 128 → 128 | 4×4 | 2 | 1 | 128 × 2 × 24 |
| ConvTranspose2D 2 | 128 → 64 | 4×4 | 2 | 1 | 64 × 4 × 48 |
| ConvTranspose2D 3 | 64 → 32 | 4×4 | 2 | 1 | 32 × 8 × 96 |
| ConvTranspose2D 4 | 32 → 22 | 4×4 | 2 | 1 | 22 × 16 × 192 |

- **Activation:** LeakyReLU with negative slope `0.2` after the first three transposed convolutional layers.
- **Normalization:** GroupNorm after the first three transposed convolutions, before LeakyReLU.
- **Final output:** 22-channel **logits** (no final sigmoid in the model when using `BCEWithLogitsLoss`).
- **Reconstruction loss:** Binary cross-entropy across all 22 channels, consistent with the proposal.
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
Input A: (22, 16, 192)              Input B: (22, 16, 192)
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
     FC 1536          |            |      FC 1536
     Reshape          |            |      Reshape
     Deconv 128       |            |      Deconv 128
     Deconv 64        |            |      Deconv 64
     Deconv 32        |            |      Deconv 32
     Deconv 22        |            |      Deconv 22
          |           |            |           |
          v           +-----+------+           v
  Reconstructed A           |          Reconstructed B
   (22, 16, 192)            v           (22, 16, 192)
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
