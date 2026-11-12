# Baseline metrics

Dataset: `preprocessed_dataset`.
Training: 232 comparisons / 464 level occurrences.
Validation: 58 comparisons / 116 level occurrences.
Each level is 16 × 176 after dropping the first 16 source columns; the validation denominator is 326,656 tile cells.

| Generated baseline | Mismatched cells | Normalized Hamming distance | Reconstruction BCE |
| --- | ---: | ---: | ---: |
| (i) All empty (`-`) | 61,744 | 0.1890 | 0.6845 |
| (ii) Bottom two rows ground (`X`), otherwise empty (`-`) | 23,715 | 0.0726 | 0.6792 |
| (iii) Most probable tile at each cell, fitted on `train_set` | 23,715 | 0.0726 | 0.6792 |

## Fun ranking: random-guessing probability baseline (p = 0.5)

These are fixed theoretical baselines; no ranking predictions or metric
calculations are needed. Reversing each comparison balances targets 0 and 1.

| Ranking baseline | Fun ranking BCE | Accuracy |
| --- | ---: | ---: |
| All ranking outputs are probability 0.5 | 0.6931 | 50% |

## Calculation

### Reconstruction BCE

Uses `binary_cross_entropy_with_logits` with the default mean reduction, as in
`loss_components` in [train.py](../../model-training/initial_model_architecture/train.py).
Each fixed grid is encoded with logit 1 for the selected tile and logit 0 for
the other 20 tile channels. All three baselines use path-channel logit 0.
These are logits: sigmoid maps 0 to probability 0.5 and 1 to approximately 0.7311.
Targets are the complete float32 binary encodings, including the path channel.
BCE is averaged over all 22 × 16 × 176 bits and both levels of each comparison,
then over validation comparisons. Averaging the per-occurrence BCE values here
is equivalent to training's `0.5 * (BCE(side_a) + BCE(side_b))` because all
occurrences have the same shape and every comparison has two levels.
The reported loss is unweighted, before applying `lambda_rec`.

### Fun ranking BCE and accuracy

Every forward and reversed ordered pair receives output probability 0.5,
independently of the three reconstruction baselines. This assigns equal
probability to both classes, giving the random-guessing BCE baseline.
For either target, `-y * log(0.5) - (1 - y) * log(0.5)` equals
`-log(0.5) = log(2)`, approximately 0.6931 (natural logarithm).
The report records this fixed, unweighted BCE directly, before `lambda_rank`.
Training's `probability >= 0.5` rule resolves these ties to class 1.
Reversing every comparison produces equal numbers of targets 0 and 1,
so exactly half of these predictions are correct: accuracy is 50%.

### Normalized Hamming distance

Matches `evaluate_training_pairs` in the same training pipeline:
take `argmax` over the first 21 tile channels, compare tile IDs at each cell,
sum mismatches across both levels in every validation comparison, and divide
by the total number of tile cells. The independent path channel (index 21)
is excluded. Tile symbols remain distinct, including `X` and `#`.
Distances range from 0 (identical) to 1 (every cell differs); lower is better.
Displayed decimals are rounded to 4 places; the counts and denominator above
give the exact ratios.

Each `.npy` occurrence has equal weight, including repeated levels in different
comparison folders, matching training evaluation. Neither split is deduplicated.
For baseline (iii), at every row and column:

```text
p(tile=t at row=r, column=c) = training occurrences with tile t at (r,c)
                              / total training level occurrences
generated_tile[r,c] = argmax_t p(tile=t at row=r, column=c)
```

Probability ties select the lowest channel index, as in NumPy/PyTorch argmax.
The same generated grid for each baseline is compared with every validation
occurrence. Only `train_set` fits the probabilities; `validation_set` supplies
the evaluation targets. No test-set encodings are read.

## Reproduce

From the repository root, using the existing training environment (NumPy and PyTorch are required):

```sh
.venv/bin/python datasets/fun-study-dataset-analysis/baseline_metrics.py
```

The script defaults to the adjacent `preprocessed_dataset` and overwrites this
report only after all calculations succeed. Optional `--dataset-root` and
`--output` arguments select other paths.
