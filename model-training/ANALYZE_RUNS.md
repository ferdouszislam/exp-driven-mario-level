# Rank saved experiment epochs

From the repository root, use the existing virtual environment:

```bash
.venv/bin/python model-training/analyze_runs.py
```

The script scans `model-training/runs/<experiment-version>/<combination>/epoch_metrics.csv`
and reads each combination's saved `config.json`. It uses the original combination
number and hyperparameters from that configuration, regardless of folder sorting
or changes to `hyperparameter-combinations.txt`. No model loading or retraining is needed.
The only extra dependency beyond Python's standard library is Matplotlib, already
listed in `model-training/requirements.txt`.

For **each experiment version separately**, it selects:

1. The 10 highest validation accuracies, breaking ties by the smaller absolute
   difference between train and validation accuracy.
2. The 10 smallest absolute train/validation accuracy differences, breaking ties
   by higher validation accuracy.

Only epochs with **both train accuracy and validation accuracy strictly greater
than 0.5** are candidates for either ranking. An epoch is excluded if either
accuracy is equal to or below 0.5.
**Multiple epochs from one combination are allowed.**
If both requested ranking keys tie, lower combination number wins, then earlier epoch.
If fewer than 10 eligible rows exist, all eligible rows are returned. If none qualify,
the CSVs contain headers only, the report states that no epochs qualify, and no PNGs
are selected for that experiment.
The difference is computed using decimal arithmetic on the logged accuracy values;
ranking does not use rounded display values or a tie tolerance.

## Outputs

By default, reports are saved under `model-training/run-analysis/`:

```text
run-analysis/
  README.md
  <experiment-version>/
    README.md
    top_validation_accuracy.csv
    smallest_accuracy_gap.csv
    plots/
      combination_01_epoch_023.png
      ...
```

Each CSV contains the rank, experiment and architecture names, original combination
number/ID and folder, learning rate, all three loss weights, batch size, latent
dimension, dropout, seeds, epoch, absolute accuracy gap, and the train/validation
values for total loss, reconstruction BCE, KL divergence, ranking BCE, accuracy,
and normalized tile Hamming distance. F1 scores are also retained when present.
Source CSV/config paths and a PNG path make each result traceable. Source paths are
relative to the input runs folder; PNG paths are relative to the experiment report.
Logged metric strings are preserved in the CSVs without rounding.

Each unique selected `(experiment version, combination, epoch)` gets **one PNG with
six bar-chart panels**, with a Train and Validation bar in every panel. If an epoch
appears in both lists, both entries link to the same PNG. Total loss is already
weighted; the three loss parts are shown unweighted, exactly as saved by training.
Accuracy and Hamming distance use the original 0–1 scale. Chart labels and Markdown
previews display six significant digits; CSVs contain the full logged values.

Rerunning overwrites matching report/PNG files. After refreshing an experiment's
reports, earlier PNGs named `combination_<number>_epoch_<number>.png` that are no
longer selected are removed from its `plots` directory. Other files are retained.
Use a different `--output-dir` to keep separate snapshots. Input run files are read only.
Malformed inputs, missing required metrics/configurations, non-finite values, and
duplicate combination numbers or epochs produce an error before report generation.
Folders with no `*/epoch_metrics.csv` are not discovered automatically; requesting
one explicitly with `--experiment` reports an error.

## Options

```bash
# Analyze just one version.
.venv/bin/python model-training/analyze_runs.py --experiment m2_e100_v3

# Analyze selected versions and keep a separate report snapshot.
.venv/bin/python model-training/analyze_runs.py \
  --experiment m1_e50 --experiment m2_e100_v2 \
  --top-n 5 --output-dir /tmp/mario-top-five

# Analyze another runs folder (paths may be relative or absolute).
.venv/bin/python model-training/analyze_runs.py \
  --runs-dir /path/to/runs --output-dir /path/to/reports
```

When `--output-dir` is omitted, `run-analysis` is created beside the selected runs
folder. Defaults work from any current directory because the default input path is
relative to the script. Output must be outside the input runs directory.

Run the focused checks with:

```bash
.venv/bin/python -m unittest discover -s model-training/tests -p 'test_analyze_runs.py'
```
