# Baseline metrics

Dataset: `preprocessed_dataset`.
Training: 232 comparisons / 464 level occurrences.
Validation: 58 comparisons / 116 level occurrences.
Each level is 16 × 176 after dropping the first 16 source columns; the validation denominator is 326,656 tile cells.

| Generated baseline | Mismatched cells | Normalized Hamming distance |
| --- | ---: | ---: |
| (i) All empty (`-`) | 61,744 | 0.1890 |
| (ii) Bottom two rows ground (`X`), otherwise empty (`-`) | 23,715 | 0.0726 |
| (iii) Most probable tile at each cell, fitted on `train_set` | 23,715 | 0.0726 |

## Calculation

Matches `evaluate_training_pairs` in [train.py](../../model-training/initial_model_architecture/train.py):
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

From the repository root, using the existing environment (NumPy is required):

```sh
.venv/bin/python datasets/fun-study-dataset-analysis/baseline_metrics.py
```

The script defaults to the adjacent `preprocessed_dataset` and overwrites this
report only after all calculations succeed. Optional `--dataset-root` and
`--output` arguments select other paths.
