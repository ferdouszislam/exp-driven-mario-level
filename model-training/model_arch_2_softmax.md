# Model architecture 2: softmax tiles and sigmoid player path

This variant inherits the layers, dimensions, normalization, hidden activations, latent sampling, FC-Tail, and **1,276,951 trainable parameters** from [model_arch_2.md](model_arch_2.md). Only reconstruction output interpretation and its training loss change. Its pipeline is [model_arch_2_softmax/](model_arch_2_softmax/README.md), and its checkpoint architecture version is `model_arch_2_crop16_softmax`.

`decode(z)` returns raw logits with shape `(batch, 22, 16, 176)`. `decode_probabilities(z)` converts them as follows:

| Channels | Prediction | Reconstruction images | Training loss |
|---|---|---|---|
| 0–20 | One tile category per cell | Softmax across the 21 tile channels, then argmax | Categorical cross-entropy against the input tile ID |
| 21 | Independent player-path occupancy | Sigmoid, then probability `>= 0.5` | Binary cross-entropy against the input path bit |

The tile ID target is `input[:, :21].argmax(dim=1)`. Softmax tile probabilities sum to 1 at each cell; the path probability is independent and can coexist with any tile. The path remains a spatial occupancy mask, with the same channel codebook and input data as the base pipeline.

Training uses `F.cross_entropy` on raw tile logits, incorporating log-softmax stably, and `F.binary_cross_entropy_with_logits` on raw path logits, incorporating sigmoid stably. Neither loss receives already activated probabilities. Each loss is averaged over all batch/grid cells, then averaged across the two playthroughs:

```text
L_tile = (mean_tile_CE_A + mean_tile_CE_B) / 2
L_path = (mean_path_BCE_A + mean_path_BCE_B) / 2
L_rec  = L_tile + L_path
L_total = lambda_rec * L_rec + lambda_kl * L_KL + lambda_rank * L_rank
```

KL and preference-ranking losses retain the base definitions. Training records `reconstruction_loss`, `tile_cross_entropy`, and `path_bce`; the base pipeline retains `reconstruction_bce`. The sum of mean tile CE and mean path BCE has a different scale and meaning from BCE averaged over all 22 bits, so reconstruction and total losses should be compared within the same objective.

Run commands, shared data handling, checkpoints, and metrics follow [TRAINING.md](TRAINING.md). This folder has its own hyperparameter file and `runs/` output location.
