# CNN-VAE + Pairwise FC-Tail Architecture Specification 1

This document specifies the revised shared CNN-VAE and concatenated-latent FC-Tail architecture. It follows the section structure of the original architecture specification and records the network implemented in `model.py`. The original **4×4 kernels are retained as an implementation assumption**. All convolutional and transposed convolutional layers use stride **2**, padding **1**, and dilation **1**. [model.py](model.py), [original architecture specification](initial_model_architecture.md).

## 1. CNN Encoder

**Input:** `(batch_size, 22, 16, 192)` in channels-first (NCHW) format.

| Layer | In → Out channels | Kernel | Stride | Padding | Output shape (C × H × W) |
|---|---|---|---|---|---|
| Conv2D 1 | 22 → 32 | 4×4 | 2 | 1 | 32 × 8 × 96 |
| Conv2D 2 | 32 → 64 | 4×4 | 2 | 1 | 64 × 4 × 48 |
| Conv2D 3 | 64 → 32 | 4×4 | 2 | 1 | 32 × 2 × 24 |
| Flatten | — | — | — | — | 1,536 |
| FC | 1,536 → 128 | — | — | — | 128 |
| FC (μ) | 128 → 32 | — | — | — | 32 |
| FC (log σ²) | 128 → 32 | — | — | — | 32 |

- **Activation:** ReLU after each convolution and after the 128-unit hidden fully connected layer.
- **Normalization:** LayerNorm after each convolution, before ReLU. The normalized shapes are `(32, 8, 96)`, `(64, 4, 48)`, and `(32, 2, 24)`.
- **LayerNorm axes:** Normalize over the full `(channel, height, width)` feature map independently for each playthrough. Use `eps=1e-5` and `elementwise_affine=True` for learned scale and bias at every normalized element. [PyTorch LayerNorm documentation](https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html).
- **Gaussian heads:** Separate affine layers return μ and log_var from the same 128-dimensional hidden representation.
- **Max pooling:** None. The strided convolutions perform downsampling.
- **Latent dimension:** `32`.
- **VAE sampling:** `z = μ + exp(0.5 * log_var) * ε`, with `ε ~ N(0, I)`.
- Each input playthrough is encoded independently using **the same encoder weights**.

For each height or width dimension, dilation 1 gives

```text
convolution_output = floor((input + 2 * padding - kernel) / stride) + 1
```

With kernel 4, padding 1, and stride 2, the encoder halves the spatial dimensions at each layer. The shapes are `16 × 192 → 8 × 96 → 4 × 48 → 2 × 24`, giving `32 × 2 × 24 = 1,536` flattened features.

## 2. CNN Decoder

**Input:** A 32-dimensional sampled latent vector (`batch_size × 32`).

| Layer | In → Out channels | Kernel | Stride | Padding | Output shape (C × H × W) |
|---|---|---|---|---|---|
| FC | 32 → 1,536 | — | — | — | 1,536 |
| Reshape | — | — | — | — | 32 × 2 × 24 |
| ConvTranspose2D 1 | 32 → 64 | 4×4 | 2 | 1 | 64 × 4 × 48 |
| ConvTranspose2D 2 | 64 → 32 | 4×4 | 2 | 1 | 32 × 8 × 96 |
| ConvTranspose2D 3 | 32 → 22 | 4×4 | 2 | 1 | 22 × 16 × 192 |

- **Activation:** ReLU after the projection FC and after the first two transposed convolutions.
- **Normalization:** LayerNorm over the full per-playthrough `(channel, height, width)` feature map after the first two transposed convolutions, before ReLU. The normalized shapes are `(64, 4, 48)` and `(32, 8, 96)`, with `eps=1e-5` and learned elementwise scale and bias.
- **Final output:** 22-channel affine logits for `BCEWithLogitsLoss`.
- **Reconstruction loss:** Binary cross-entropy across all 22 channels and all grid cells, averaged over both input playthroughs.
- **Max pooling/unpooling:** None.
- The decoder is shared between both playthroughs.

With dilation 1 and output padding 0, the transposed convolution dimension is

```text
transposed_convolution_output = (input - 1) * stride - 2 * padding + kernel
```

The decoder doubles the spatial dimensions at each transposed convolution. Its shapes are `2 × 24 → 4 × 48 → 8 × 96 → 16 × 192`.

## 3. FC-Tail: Pairwise Preference Predictor

**Input:** Concatenation of two 32-dimensional sampled latent vectors in the given input order.

`[z_A; z_B]` has shape `(batch_size, 64)`.

| Layer | Input size | Output size | Activation |
|---|---:|---:|---|
| FC 1 | 64 | 64 | ReLU |
| Dropout 1 | 64 | 64 | Dropout(0.2) |
| FC 2 | 64 | 32 | ReLU |
| Dropout 2 | 32 | 32 | Dropout(0.2) |
| FC 3 | 32 | 1 | Linear logit |

- **Output:** Sigmoid of the final logit gives `p(z_A, z_B)`, the probability that the **first** playthrough is preferred to the second.
- **Preference loss:** Binary cross-entropy with logits against the ordered pairwise target.
- **Hidden activations:** ReLU, with dropout probability `0.2` after each of the two hidden layers. Dropout is active during optimization and disabled during evaluation.
- **Final layer:** An affine scalar output. Sigmoid is applied when obtaining probabilities.
- Use the **same FC-Tail** for all ordered pairs.

## 4. Overall Architecture

```text
Input A: (22, 16, 192)              Input B: (22, 16, 192)
          |                                    |
          v                                    v
     SHARED CNN ENCODER                  SHARED CNN ENCODER
     Conv 22->32                         Conv 22->32
     Conv 32->64                         Conv 32->64
     Conv 64->32                         Conv 64->32
     Flatten -> FC 128                   Flatten -> FC 128
     mu: 32, log_var: 32                mu: 32, log_var: 32
          |                                    |
          v                                    v
        z_A: 32                              z_B: 32
          |                                    |
          +-----------+------------+-----------+
          |           |            |           |
          v           |            |           v
     SHARED CNN       |            |      SHARED CNN
       DECODER        |            |        DECODER
     FC 1,536         |            |      FC 1,536
     Reshape          |            |      Reshape
     32 x 2 x 24      |            |      32 x 2 x 24
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
                    Logit / Sigmoid
                           |
                           v
                      P(A preferred)
```

### Architecture summary

- **Encoder:** 3 convolutional layers, 1 hidden FC layer, and 2 affine Gaussian parameter heads.
- **Decoder:** 1 projection FC layer, reshape, and 3 transposed convolutional layers.
- **Latent space:** 32 dimensions per playthrough, with reparameterized sampling.
- **Activations:** ReLU after all three encoder convolutions, the encoder hidden FC, the decoder projection, the first two decoder transposed convolutions, and both FC-Tail hidden layers.
- **Normalization:** LayerNorm over `(channel, height, width)` after all three encoder convolutions and the first two decoder transposed convolutions.
- **FC-Tail:** 3 fully connected layers (`64 → 64 → 32 → 1`), with Dropout(0.2) after both hidden layers.
- **Pooling:** No max-pooling or unpooling.
- **Pair processing:** Two independent passes through the same encoder, followed by concatenation of the latent samples in input order.
- **Reconstruction:** Each latent sample is decoded by the same decoder into 22-channel logits.
- **Joint training loss:** `L = λ_rec * L_rec + λ_KL * L_KL + λ_rank * L_rank`.

`L_rec` averages reconstruction BCE over all 22 channels, all 16×192 positions, the batch, and both playthroughs. `L_KL` sums each posterior's standard-normal KL divergence over its 32 latent coordinates, then averages over the batch and both playthroughs. `L_rank` averages binary preference BCE over ordered comparisons, with target 1 for preferred-first and 0 for reversed input order.

Reconstruction gradients update the shared encoder and decoder. KL gradients update the encoder and Gaussian heads. Ranking gradients pass through the FC-Tail and both reparameterized samples to the shared encoder. The training guide specifies the full reductions and experiment settings. [TRAINING.md](TRAINING.md).

### Trainable parameter counts

The network contains **566,327 trainable parameters**. Shared modules are counted once.

| Component | Trainable parameters |
|---|---:|
| Encoder, including both Gaussian heads | 358,720 |
| Decoder | 201,334 |
| Pairwise FC-Tail | 6,273 |
| **Total** | **566,327** |

| Component | Layer | Parameters |
|---|---|---:|
| Encoder | Conv 22 → 32 and LayerNorm((32, 8, 96)) | 60,448 |
| Encoder | Conv 32 → 64 and LayerNorm((64, 4, 48)) | 57,408 |
| Encoder | Conv 64 → 32 and LayerNorm((32, 2, 24)) | 35,872 |
| Encoder | Hidden FC 1,536 → 128 | 196,736 |
| Encoder | Mean head 128 → 32 | 4,128 |
| Encoder | Log-variance head 128 → 32 | 4,128 |
| Decoder | Projection FC 32 → 1,536 | 50,688 |
| Decoder | Transposed conv 32 → 64 and LayerNorm((64, 4, 48)) | 57,408 |
| Decoder | Transposed conv 64 → 32 and LayerNorm((32, 8, 96)) | 81,952 |
| Decoder | Output transposed conv 32 → 22 | 11,286 |
| FC-Tail | Hidden FC 64 → 64 | 4,160 |
| FC-Tail | Hidden FC 64 → 32 | 2,080 |
| FC-Tail | Output FC 32 → 1 | 33 |

These counts include affine biases and LayerNorm's learned scale and shift. Each LayerNorm contributes `2 × C × H × W` parameters for its normalized feature-map shape. ReLU, dropout, reshaping, flattening, concatenation, sigmoid, and latent sampling contribute zero parameters.

### Checkpoint architecture metadata

New checkpoints record `architecture_version="model_arch_1"`, `normalization="LayerNorm"`, `layer_norm_axes=["channel","height","width"]`, `stride=[2,2]`, `encoded_shape=[32,2,24]`, and `flattened_features=1536`. Metadata also records encoder channels `[22,32,64,32]`, decoder channels `[32,64,32,22]`, and the normalized feature-map shapes. Check the full normalization, channel, stride, and shape metadata together with the architecture label before restoring a checkpoint with the current `PairwiseCNNVAE` class.

`encoder_layer_norm_shapes` records `[[32,8,96],[64,4,48],[32,2,24]]`, and `decoder_layer_norm_shapes` records `[[64,4,48],[32,8,96]]`. The normalization settings are saved as `layer_norm_eps=1e-5` and `layer_norm_elementwise_affine=True`.

Earlier two-convolution checkpoints used the same architecture label with different shapes and normalization. Those checkpoints and the original four-convolution checkpoints require their corresponding model definitions for restoration. The original specification records the four-convolution starting architecture. [initial_model_architecture.md](initial_model_architecture.md).
