# Training results

This run uses only the selected training comparisons.

- Comparisons: 4 of 290
- Ordered examples: 8 (both orders, balanced labels)
- Final fitting epochs: 2
- Device: cpu
- Hyperparameter selection: completed
- Hyperparameter selection rule: smallest absolute fitting/validation total-loss gap; ties use candidate order
- Adam learning rate: 0.001
- Weights (reconstruction, KL, ranking): 1.0, 0.01, 1.0

The selected tuning candidate was measured in evaluation mode on both portions:

| Tuning comparison | Value |
|---|---:|
| Fitting total loss | 1.3766856194 |
| Validation total loss | 1.3817589283 |
| Validation minus fitting total loss | 0.0050733089 |
| Absolute total-loss gap (selection score) | 0.0050733089 |

This gap belongs to the tuning candidate before fresh final training.
A small gap can also occur when both losses are high; it does not guarantee
that the model avoids overfitting.

| Final train measurement | Value |
|---|---:|
| Weighted total loss | 1.3341753483 |
| Reconstruction BCE (mean per bit) | 0.6323614717 |
| KL divergence (sum per latent, mean per input) | 0.7876509428 |
| Fun-ranking BCE | 0.6939374208 |
| Fun-ranking accuracy | 0.5000000000 |
| Fun-ranking macro F1 | 0.3333333333 |

Final measurements use evaluation mode and one seeded posterior draw per input;
the same two latent samples are reused for the two orders. These are in-sample
training results, not estimates of held-out performance. No test examples are loaded.

Confusion matrix (true rows, predicted columns; labels 0, 1): `[[0, 4], [0, 4]]`.

`model.pt` contains model and Adam states, architecture, codebook, selected IDs,
configuration, dataset fingerprint and final train metrics. `train_predictions.csv`
contains every ordered prediction. See `tuning.json` and `train_history.json` for
selection details and stochastic optimization losses.
