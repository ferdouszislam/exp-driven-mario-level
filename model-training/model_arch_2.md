# CNN-VAE + Pairwise FC-Tail Architecture Specification 2

This document specifies architecture **`model_arch_2`**, implemented in [model.py](model.py). It retains the four-stage **shared CNN-VAE** and **concatenated-latent FC-Tail** from the [original architecture specification](initial_model_architecture.md), using LayerNorm, ReLU, and Dropout(0.2) after the first FC-Tail hidden layer. Training and checkpoint previews are described in [TRAINING.md](TRAINING.md).

## 1. CNN Encoder

**Input:** `(batch_size, 22, 16, 192)` in channels-first (NCHW) format. Channels 0–20 encode tile identity; channel 21 independently encodes the recorded player path.

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

- **Activation:** ReLU after each convolution and after the 128-unit hidden fully connected layer.
- **Normalization:** LayerNorm after each convolution, before ReLU. The normalized shapes are `(32, 8, 96)`, `(64, 4, 48)`, `(128, 2, 24)`, and `(128, 1, 12)`.
- **LayerNorm axes:** The complete `(channel, height, width)` feature map, independently for each playthrough, with `eps=1e-5` and learned elementwise scale and bias.
- **Gaussian heads:** Separate linear outputs for μ and log_var from the shared 128-dimensional hidden representation.
- **Max pooling:** None. The strided convolutions perform downsampling.
- **Latent dimension:** `32`.
- **VAE sampling (reparameterization):** `z = μ + exp(0.5 * log_var) * ε`, with `ε ~ N(0, I)`.
- Each input playthrough is encoded independently using **the same encoder weights**.

## 2. CNN Decoder

**Input:** A 32-dimensional sampled latent vector (`batch_size × 32`).

| Layer | In → Out channels | Kernel | Stride | Padding | Output shape (C × H × W) |
|---|---|---|---|---|---|
| FC | 32 → 1,536 | — | — | — | 1,536 |
| Reshape | — | — | — | — | 128 × 1 × 12 |
| ConvTranspose2D 1 | 128 → 128 | 4×4 | 2 | 1 | 128 × 2 × 24 |
| ConvTranspose2D 2 | 128 → 64 | 4×4 | 2 | 1 | 64 × 4 × 48 |
| ConvTranspose2D 3 | 64 → 32 | 4×4 | 2 | 1 | 32 × 8 × 96 |
| ConvTranspose2D 4 | 32 → 22 | 4×4 | 2 | 1 | 22 × 16 × 192 |

- **Projection FC:** Linear, followed directly by reshaping; no activation is applied to the projection.
- **Activation:** ReLU after the first three transposed convolutional layers.
- **Normalization:** LayerNorm after the first three transposed convolutions, before ReLU. The normalized shapes are `(128, 2, 24)`, `(64, 4, 48)`, and `(32, 8, 96)`.
- **LayerNorm settings:** Normalize over the complete per-playthrough feature map with `eps=1e-5` and learned elementwise scale and bias.
- **Final output:** 22-channel **logits**, with no final sigmoid in the model.
- **Reconstruction loss:** Binary cross-entropy with logits across all 22 channels.
- **Max pooling/unpooling:** None.
- **Convolution settings:** Dilation `1`; transposed-convolution output padding `0`.
- The decoder is shared between both playthroughs.

## 3. FC-Tail: Pairwise Preference Predictor

**Input:** Concatenation of two 32-dimensional sampled latent vectors in the given input order.

`[z_A; z_B]` has shape `(batch_size, 64)`.

| Layer | Input size | Output size | Activation |
|---|---:|---:|---|
| FC 1 | 64 | 64 | ReLU |
| Dropout | 64 | 64 | Dropout(0.2) |
| FC 2 | 64 | 32 | ReLU |
| FC 3 | 32 | 1 | Linear logit |

- **Output:** Sigmoid of the final logit gives `p(z_A, z_B)`, the probability that the **first** playthrough is preferred to the second.
- **Preference loss:** Binary cross-entropy with logits against the ordered pairwise target.
- **Dropout:** Probability `0.2`, only after the first hidden FC layer. It is active during optimization and disabled during evaluation.
- **Final layer:** A linear scalar logit; sigmoid is applied only when obtaining probabilities.
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
     128 x 1 x 12     |            |      128 x 1 x 12
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
                    ReLU, Dropout(0.2)
                       64 -> 32
                         ReLU
                       32 -> 1
                     Linear logit
                           |
                           v
                  Sigmoid for probability
                      P(A preferred)
```

### Architecture summary

- **Encoder:** 4 convolutional layers, 1 shared hidden FC layer, and 2 Gaussian parameter heads.
- **Decoder:** 1 linear projection FC layer, reshape, and 4 transposed convolutional layers.
- **Latent space:** 32 dimensions per playthrough.
- **Activation:** ReLU after every encoder convolution, the encoder hidden FC, the first three decoder convolutions, and both FC-Tail hidden layers.
- **Normalization:** LayerNorm over `(channel, height, width)` after all four encoder convolutions and the first three decoder transposed convolutions.
- **FC-Tail:** 3 fully connected layers (`64 → 64 → 32 → 1`), with Dropout(0.2) after the first hidden layer.
- **Pooling:** No max-pooling or unpooling.
- **Pair processing:** Two independent passes through the **same** encoder; latent samples are concatenated in input order for pairwise classification.
- **Reconstruction:** Each latent sample is decoded by the **same** decoder into 22-channel logits.
- **Joint training loss:** `L = λ_rec * L_rec + λ_KL * L_KL + λ_rank * L_rank`.

`L_rec` averages reconstruction BCE over all 22 channels, all 16×192 positions, the batch, and both playthroughs. `L_KL` sums each posterior's standard-normal KL divergence over its 32 latent coordinates, then averages over the batch and both playthroughs. `L_rank` averages binary preference BCE over ordered comparisons, with target 1 for preferred-first and 0 for reversed input order. The coefficient values come from the editable hyperparameter file, separately from this fixed architecture.

Reconstruction gradients update the shared encoder and decoder. KL gradients update the encoder and Gaussian heads. Ranking gradients pass through the FC-Tail and both reparameterized samples to the shared encoder. The training guide specifies the full reductions, experiment settings, and checkpoint previews. [TRAINING.md](TRAINING.md).

### Trainable parameter counts

The network contains **1,312,151 trainable parameters**. Shared modules are counted once.

| Component | Trainable parameters |
|---|---:|
| Encoder, including both Gaussian heads | 731,680 |
| Decoder | 574,198 |
| Pairwise FC-Tail | 6,273 |
| **Total** | **1,312,151** |

| Component | Layer | Parameters |
|---|---|---:|
| Encoder | Conv 22 → 32 and LayerNorm((32, 8, 96)) | 60,448 |
| Encoder | Conv 32 → 64 and LayerNorm((64, 4, 48)) | 57,408 |
| Encoder | Conv 64 → 128 and LayerNorm((128, 2, 24)) | 143,488 |
| Encoder | Conv 128 → 128 and LayerNorm((128, 1, 12)) | 265,344 |
| Encoder | Hidden FC 1,536 → 128 | 196,736 |
| Encoder | Mean head 128 → 32 | 4,128 |
| Encoder | Log-variance head 128 → 32 | 4,128 |
| Decoder | Projection FC 32 → 1,536 | 50,688 |
| Decoder | Transposed conv 128 → 128 and LayerNorm((128, 2, 24)) | 274,560 |
| Decoder | Transposed conv 128 → 64 and LayerNorm((64, 4, 48)) | 155,712 |
| Decoder | Transposed conv 64 → 32 and LayerNorm((32, 8, 96)) | 81,952 |
| Decoder | Output transposed conv 32 → 22 | 11,286 |
| FC-Tail | Hidden FC 64 → 64 | 4,160 |
| FC-Tail | Hidden FC 64 → 32 | 2,080 |
| FC-Tail | Output FC 32 → 1 | 33 |

These counts include affine biases and LayerNorm's learned scale and shift. Each LayerNorm contributes `2 × C × H × W` parameters for its normalized feature-map shape. ReLU, dropout, reshaping, flattening, concatenation, sigmoid, and latent sampling contribute zero parameters.

### Checkpoint architecture metadata

New checkpoints record `architecture_version="model_arch_2"`, `normalization="LayerNorm"`, `layer_norm_axes=["channel","height","width"]`, `stride=[2,2]`, `encoded_shape=[128,1,12]`, and `flattened_features=1536`. Metadata also records encoder channels `[22,32,64,128,128]` and decoder channels `[128,128,64,32,22]`.

`encoder_layer_norm_shapes` records `[[32,8,96],[64,4,48],[128,2,24],[128,1,12]]`, and `decoder_layer_norm_shapes` records `[[128,2,24],[64,4,48],[32,8,96]]`. The normalization settings are saved as `layer_norm_eps=1e-5` and `layer_norm_elementwise_affine=True`. Activation metadata records ReLU, a linear decoder projection, dropout probability `0.2`, and dropout placement `[1]`.

Existing checkpoints retain their saved version labels. Check the full normalization, channel, stride, shape, activation, and dropout metadata when choosing a matching model definition for restoration.

**Status:** These layer dimensions are implementation choices, not exact configurations claimed to be taken from a cited paper.
