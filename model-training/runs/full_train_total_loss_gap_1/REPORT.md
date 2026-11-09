# Training results

This run uses only the selected training comparisons.

- Comparisons: 290 of 290
- Ordered examples: 580 (both orders, balanced labels)
- Final fitting epochs: 50
- Device: mps
- Hyperparameter selection: completed
- Hyperparameter selection rule: smallest absolute fitting/validation total-loss gap; ties use candidate order
- Adam learning rate: 0.001
- Weights (reconstruction, KL, ranking): 1.0, 0.01, 1.0

The selected tuning candidate was measured in evaluation mode on both portions:

| Tuning comparison | Value |
|---|---:|
| Fitting total loss | 0.7505305451 |
| Validation total loss | 0.7529270197 |
| Validation minus fitting total loss | 0.0023964746 |
| Absolute total-loss gap (selection score) | 0.0023964746 |

This gap belongs to the tuning candidate before fresh final training.
A small gap can also occur when both losses are high; it does not guarantee
that the model avoids overfitting.

| Final train measurement | Value |
|---|---:|
| Weighted total loss | 0.0875454369 |
| Reconstruction BCE (mean per bit) | 0.0328717198 |
| KL divergence (sum per latent, mean per input) | 3.6076908506 |
| Fun-ranking BCE | 0.0185968083 |
| Fun-ranking accuracy | 0.9965517241 |
| Fun-ranking F1 (positive class 1) | 0.9965517241 |

Final measurements use evaluation mode and one seeded posterior draw per input;
the same two latent samples are reused for the two orders. These are in-sample
training results, not estimates of held-out performance. No test examples are loaded.

Confusion matrix (true rows, predicted columns; labels 0, 1): `[[289, 1], [1, 289]]`.

`model.pt` contains model and Adam states, architecture, codebook, selected IDs,
configuration, dataset fingerprint and final train metrics. `train_predictions.csv`
contains every ordered prediction. See `tuning.json` and `train_history.json` for
selection details and stochastic optimization losses.
