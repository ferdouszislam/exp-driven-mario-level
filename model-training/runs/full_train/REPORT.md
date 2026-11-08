# Training results

This run uses only the selected training comparisons.

- Comparisons: 290 of 290
- Ordered examples: 580 (both orders, balanced labels)
- Final fitting epochs: 50
- Device: cpu
- Hyperparameter selection: completed
- Adam learning rate: 0.001
- Weights (reconstruction, KL, ranking): 1.0, 0.01, 1.0

| Final train measurement | Value |
|---|---:|
| Weighted total loss | 0.1143303096 |
| Reconstruction BCE (mean per bit) | 0.0328517547 |
| KL divergence (sum per latent, mean per input) | 4.2113034610 |
| Fun-ranking BCE | 0.0393655205 |
| Fun-ranking accuracy | 0.9913793103 |
| Fun-ranking macro F1 | 0.9913792847 |

Final measurements use evaluation mode and one seeded posterior draw per input;
the same two latent samples are reused for the two orders. These are in-sample
training results, not estimates of held-out performance. No test examples are loaded.

Confusion matrix (true rows, predicted columns; labels 0, 1): `[[287, 3], [2, 288]]`.

`model.pt` contains model and Adam states, architecture, codebook, selected IDs,
configuration, dataset fingerprint and final train metrics. `train_predictions.csv`
contains every ordered prediction. See `tuning.json` and `train_history.json` for
selection details and stochastic optimization losses.
