#!/usr/bin/env python3
"""Score reconstruction baselines and report fixed theoretical ranking baselines.

Run from the repository root:
    .venv/bin/python datasets/fun-study-dataset-analysis/baseline_metrics.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
TILE_SYMBOLS = list("XS-%tToQ@UL12gkrKRy#|")
SHAPE = (22, 16, 176)
TILE_CHANNELS = 21


def load_encoding(path: Path) -> np.ndarray:
    """Validate the same binary tile/path representation used by training."""
    encoding = np.load(path, allow_pickle=False)
    if encoding.shape != SHAPE or encoding.dtype != np.uint8:
        raise ValueError(f"Expected uint8 {SHAPE} encoding: {path}")
    if not np.all((encoding == 0) | (encoding == 1)):
        raise ValueError(f"Expected binary encoding: {path}")
    if not np.all(encoding[:TILE_CHANNELS].sum(axis=0) == 1):
        raise ValueError(f"Expected exactly one tile per cell: {path}")
    return encoding


def split_files(root: Path, split: str, manifest: dict) -> tuple[list[Path], int]:
    """Keep both sides of every comparison, including repeated level IDs."""
    entries = [entry for entry in manifest["instances"] if entry["split"] == split]
    if not entries:
        raise ValueError(f"No {split} comparisons in manifest.json")
    files = []
    for entry in sorted(entries, key=lambda item: item["instance_id"]):
        plays = entry["playthroughs"]
        expected = {
            f"{split}/{entry['instance_id']}/{entry[field]}.npy"
            for field in ("preferred_level_id", "less_preferred_level_id")
        }
        actual = {play["files"]["encoding"] for play in plays}
        if len(plays) != 2 or len(expected) != 2 or actual != expected:
            raise ValueError(f"Expected both comparison encodings: {entry['instance_id']}")
        files.extend(root / name for name in sorted(actual))
    if len(set(files)) != len(files) or set(files) != set((root / split).glob("*/*.npy")):
        raise ValueError(f"{split} encodings do not match manifest.json")
    return files, len(entries)


def calculate_baselines(root: Path) -> dict:
    """Fit cell probabilities on train_set and score only validation_set."""
    channels = json.loads((root / "channels.json").read_text(encoding="utf-8"))
    if (channels["shape"] != list(SHAPE)
            or channels["axis_order"] != ["channel", "row", "column"]
            or channels["path_channel_index"] != TILE_CHANNELS
            or [channel["symbol"] for channel in channels["channels"]] != TILE_SYMBOLS + ["P"]):
        raise ValueError("channels.json does not match the training tile/path encoding")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    train_files, train_comparisons = split_files(root, "train_set", manifest)
    validation_files, validation_comparisons = split_files(root, "validation_set", manifest)

    counts = np.zeros((TILE_CHANNELS, *SHAPE[1:]), dtype=np.int64)
    for path in train_files:
        counts += load_encoding(path)[:TILE_CHANNELS]
    probabilities = counts / len(train_files)

    empty = np.full(SHAPE[1:], TILE_SYMBOLS.index("-"), dtype=np.int64)
    ground = empty.copy()
    ground[-2:, :] = TILE_SYMBOLS.index("X")
    baselines = {
        "(i) All empty (`-`)": empty,
        "(ii) Bottom two rows ground (`X`), otherwise empty (`-`)": ground,
        "(iii) Most probable tile at each cell, fitted on `train_set`": probabilities.argmax(axis=0),
    }
    # Treat generated binary tile encodings as logits. Keep the path logit
    # at zero so every baseline has the full 22-channel training output shape.
    baseline_logits = {}
    for name, generated_tiles in baselines.items():
        logits = torch.zeros(SHAPE, dtype=torch.float32)
        logits[:TILE_CHANNELS] = F.one_hot(
            torch.from_numpy(generated_tiles), num_classes=TILE_CHANNELS,
        ).permute(2, 0, 1).float()
        baseline_logits[name] = logits

    mismatches = {name: 0 for name in baselines}
    reconstruction_totals = {name: 0.0 for name in baselines}
    total_tiles = 0
    for path in validation_files:
        encoding = load_encoding(path)
        input_tiles = encoding[:TILE_CHANNELS].argmax(axis=0)
        target = torch.from_numpy(encoding.astype(np.float32))
        for name, generated_tiles in baselines.items():
            mismatches[name] += int(np.count_nonzero(input_tiles != generated_tiles))
            reconstruction_totals[name] += F.binary_cross_entropy_with_logits(
                baseline_logits[name], target,
            ).item()
        total_tiles += input_tiles.size

    return {
        "train_comparisons": train_comparisons,
        "train_occurrences": len(train_files),
        "validation_comparisons": validation_comparisons,
        "validation_occurrences": len(validation_files),
        "total_tiles": total_tiles,
        "mismatches": mismatches,
        "normalized_hamming_distance": {
            name: count / total_tiles for name, count in mismatches.items()
        },
        "reconstruction_bce": {
            name: total / len(validation_files) for name, total in reconstruction_totals.items()
        },
    }


def render_report(results: dict, root: Path) -> str:
    try:
        dataset_path = root.resolve().relative_to(HERE).as_posix()
    except ValueError:
        dataset_path = str(root.resolve())
    lines = [
        "# Baseline metrics",
        "",
        f"Dataset: `{dataset_path}`.",
        f"Training: {results['train_comparisons']} comparisons / "
        f"{results['train_occurrences']} level occurrences.",
        f"Validation: {results['validation_comparisons']} comparisons / "
        f"{results['validation_occurrences']} level occurrences.",
        f"Each level is {SHAPE[1]} × {SHAPE[2]} after dropping the first 16 source columns; "
        f"the validation denominator is "
        f"{results['total_tiles']:,} tile cells.",
        "",
        "| Generated baseline | Mismatched cells | Normalized Hamming distance | Reconstruction BCE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, count in results["mismatches"].items():
        distance = results["normalized_hamming_distance"][name]
        reconstruction = results["reconstruction_bce"][name]
        lines.append(f"| {name} | {count:,} | {distance:.4f} | {reconstruction:.4f} |")
    lines += [
        "",
        "## Fun ranking: random-guessing probability baseline (p = 0.5)",
        "",
        "These are fixed theoretical baselines; no ranking predictions or metric",
        "calculations are needed. Reversing each comparison balances targets 0 and 1.",
        "",
        "| Ranking baseline | Fun ranking BCE | Accuracy |",
        "| --- | ---: | ---: |",
        "| All ranking outputs are probability 0.5 | 0.6931 | 50% |",
        "",
        "## Calculation",
        "",
        "### Reconstruction BCE",
        "",
        "Uses `binary_cross_entropy_with_logits` with the default mean reduction, as in",
        "`loss_components` in [train.py](../../model-training/initial_model_architecture/train.py).",
        "Each fixed grid is encoded with logit 1 for the selected tile and logit 0 for",
        "the other 20 tile channels. All three baselines use path-channel logit 0.",
        "These are logits: sigmoid maps 0 to probability 0.5 and 1 to approximately 0.7311.",
        "Targets are the complete float32 binary encodings, including the path channel.",
        "BCE is averaged over all 22 × 16 × 176 bits and both levels of each comparison,",
        "then over validation comparisons. Averaging the per-occurrence BCE values here",
        "is equivalent to training's `0.5 * (BCE(side_a) + BCE(side_b))` because all",
        "occurrences have the same shape and every comparison has two levels.",
        "The reported loss is unweighted, before applying `lambda_rec`.",
        "",
        "### Fun ranking BCE and accuracy",
        "",
        "Every forward and reversed ordered pair receives output probability 0.5,",
        "independently of the three reconstruction baselines. This assigns equal",
        "probability to both classes, giving the random-guessing BCE baseline.",
        "For either target, `-y * log(0.5) - (1 - y) * log(0.5)` equals",
        "`-log(0.5) = log(2)`, approximately 0.6931 (natural logarithm).",
        "The report records this fixed, unweighted BCE directly, before `lambda_rank`.",
        "Training's `probability >= 0.5` rule resolves these ties to class 1.",
        "Reversing every comparison produces equal numbers of targets 0 and 1,",
        "so exactly half of these predictions are correct: accuracy is 50%.",
        "",
        "### Normalized Hamming distance",
        "",
        "Matches `evaluate_training_pairs` in the same training pipeline:",
        "take `argmax` over the first 21 tile channels, compare tile IDs at each cell,",
        "sum mismatches across both levels in every validation comparison, and divide",
        "by the total number of tile cells. The independent path channel (index 21)",
        "is excluded. Tile symbols remain distinct, including `X` and `#`.",
        "Distances range from 0 (identical) to 1 (every cell differs); lower is better.",
        "Displayed decimals are rounded to 4 places; the counts and denominator above",
        "give the exact ratios.",
        "",
        "Each `.npy` occurrence has equal weight, including repeated levels in different",
        "comparison folders, matching training evaluation. Neither split is deduplicated.",
        "For baseline (iii), at every row and column:",
        "",
        "```text",
        "p(tile=t at row=r, column=c) = training occurrences with tile t at (r,c)",
        "                              / total training level occurrences",
        "generated_tile[r,c] = argmax_t p(tile=t at row=r, column=c)",
        "```",
        "",
        "Probability ties select the lowest channel index, as in NumPy/PyTorch argmax.",
        "The same generated grid for each baseline is compared with every validation",
        "occurrence. Only `train_set` fits the probabilities; `validation_set` supplies",
        "the evaluation targets. No test-set encodings are read.",
        "",
        "## Reproduce",
        "",
        "From the repository root, using the existing training environment (NumPy and PyTorch are required):",
        "",
        "```sh",
        ".venv/bin/python datasets/fun-study-dataset-analysis/baseline_metrics.py",
        "```",
        "",
        "The script defaults to the adjacent `preprocessed_dataset` and overwrites this",
        "report only after all calculations succeed. Optional `--dataset-root` and",
        "`--output` arguments select other paths.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=HERE / "preprocessed_dataset")
    parser.add_argument("--output", type=Path, default=HERE / "baseline-metrics.md")
    args = parser.parse_args()
    results = calculate_baselines(args.dataset_root)
    report = render_report(results, args.dataset_root)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
