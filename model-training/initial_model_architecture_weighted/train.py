"""Train initial_model_architecture_weighted with a configurable air/ground error penalty."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model import PairwiseCNNVAE, architecture_metadata

# Keep this folder's model local while reusing the common pipeline utilities.
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from background_error_weighting import weighted_bce_reconstruction
from checkpoint_ranges import (
    add_checkpoint_arguments, checkpoint_epoch_selected, checkpoint_summary,
    validate_checkpoint_ranges,
)
from data import ComparisonDataset, load_training_data
from plotting import plot_loss_curves
from reconstruction_previews import (
    PREVIEW_COUNT, save_checkpoint_reconstructions, select_validation_levels,
)


# Set this to a small positive number for a quick run; None uses all train pairs.
# Counts original comparison folders BEFORE generating both input orders.
# Whole comparison groups are retained, so the actual count can be below this cap.
NUM_TRAINING_SAMPLES: int | None = None
EPOCHS = 100
# Set inclusive bounds to integers or paired lists (e.g. [10, 50] and [20, 60]).
CHECKPOINT_START_EPOCH: int | list[int] | None = None
CHECKPOINT_END_EPOCH: int | list[int] | None = None
BATCH_SIZE = 16
SEED = 42
HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE.parent.parent / "datasets/fun-study-dataset-analysis/preprocessed_dataset"
DEFAULT_HYPERPARAMETERS = HERE / "hyperparameter-combinations.txt"


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    lambda_rec: float
    lambda_kl: float
    lambda_rank: float
    background_error_weight: float


LOSS_NAMES = ("total_loss", "reconstruction_bce", "kl_divergence", "ranking_bce")


def hyperparameter_directory_name(hp: Hyperparameters) -> str:
    """Name a run by learning rate, reconstruction, KL, ranking, then background error weight."""
    def format_value(value: float) -> str:
        number = Decimal(str(value)).normalize()
        if number == 0:
            return "0"
        if abs(number) < 1:
            coefficient, exponent = format(number, "e").split("e")
            return f"{coefficient}e{int(exponent)}"
        return format(number, "f")

    return "_".join(format_value(value) for value in (
        hp.learning_rate, hp.lambda_rec, hp.lambda_kl, hp.lambda_rank, hp.background_error_weight,
    ))


def load_hyperparameter_combinations(path: Path) -> tuple[Hyperparameters, ...]:
    """Read learning-rate, reconstruction, KL, ranking, background-error-weight rows."""
    names = ("learning_rate", "lambda_rec", "lambda_kl", "lambda_rank", "background_error_weight")
    combinations = []
    seen = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        location = f"{path}:{line_number}"
        columns = line.split(",")
        if len(columns) != len(names):
            raise ValueError(f"{location}: Expected five comma-separated values: {', '.join(names)}.")
        try:
            values = tuple(float(value.strip()) for value in columns)
        except ValueError as error:
            raise ValueError(f"{location}: Each combination needs five numeric values: {', '.join(names)}.") from error
        for name, value in zip(names, values, strict=True):
            if not math.isfinite(value) or value < 0 or (name == "learning_rate" and value == 0):
                requirement = "positive" if name == "learning_rate" else "non-negative"
                raise ValueError(f"{location}: {name} must be finite and {requirement}.")
            if name == "background_error_weight" and value < 1:
                raise ValueError(f"{location}: background_error_weight must be finite and at least 1.")
        hp = Hyperparameters(*values)
        if hp in seen:
            raise ValueError(f"{location}: Duplicate hyperparameter combination.")
        seen.add(hp)
        combinations.append(hp)
    if not combinations:
        raise ValueError(f"{path}: Specify at least one hyperparameter combination.")
    return tuple(combinations)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    return torch.device(requested)


@contextmanager
def preserve_random_state():
    """Keep seeded reporting passes from resetting the training random stream."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def loss_components(output: dict, x_a: torch.Tensor, x_b: torch.Tensor,
                    target: torch.Tensor, background_error_weight: float = 1.0,
                    ) -> dict[str, torch.Tensor]:
    """Weight tile BCE at wrong air/ground predictions, preserving the 22-bit mean."""
    reconstruction = 0.5 * (
        weighted_bce_reconstruction(
            output["reconstruction_logits_a"], x_a, background_error_weight,
        )
        + weighted_bce_reconstruction(
            output["reconstruction_logits_b"], x_b, background_error_weight,
        )
    )
    divergences = []
    for side in ("a", "b"):
        mu, log_var = output[f"mu_{side}"], output[f"log_var_{side}"]
        divergences.append(-0.5 * (1 + log_var - mu.square() - log_var.exp()).sum(1).mean())
    return {
        "reconstruction_bce": reconstruction,
        "kl_divergence": 0.5 * (divergences[0] + divergences[1]),
        "ranking_bce": F.binary_cross_entropy_with_logits(output["rank_logits"], target),
    }


def weighted_loss(parts: dict, hp: Hyperparameters) -> torch.Tensor:
    return (hp.lambda_rec * parts["reconstruction_bce"]
            + hp.lambda_kl * parts["kl_divergence"]
            + hp.lambda_rank * parts["ranking_bce"])


def classification_metrics(targets: list[int], predictions: list[int]) -> dict:
    matrix = [[0, 0], [0, 0]]
    for target, predicted in zip(targets, predictions, strict=True):
        matrix[target][predicted] += 1
    if not targets:
        raise ValueError("Cannot compute metrics without observations.")
    tp, fp, fn = matrix[1][1], matrix[0][1], matrix[1][0]
    denominator = 2 * tp + fp + fn
    return {
        "accuracy": (matrix[0][0] + matrix[1][1]) / len(targets),
        "f1_score": 2 * tp / denominator if denominator else 0.0,
        "f1_positive_label": 1,
        "confusion_matrix": matrix,
        "confusion_matrix_axes": "rows=true, columns=predicted; labels=[0,1]",
        "class_support": {"0": sum(matrix[0]), "1": sum(matrix[1])},
        "ordered_examples": len(targets),
    }


def make_loader(data, entries, batch_size: int, *, ordered: bool, shuffle: bool,
                seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(ComparisonDataset(data, entries, ordered=ordered),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=generator)


def train_one_epoch(model: PairwiseCNNVAE, loader: DataLoader,
                    optimizer: torch.optim.Optimizer, hp: Hyperparameters,
                    device: torch.device) -> dict[str, float]:
    model.train()
    totals = {name: 0.0 for name in LOSS_NAMES}
    count = 0
    for x_a, x_b, target in loader:
        x_a, x_b, target = x_a.to(device), x_b.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        parts = loss_components(
            model(x_a, x_b), x_a, x_b, target, hp.background_error_weight,
        )
        loss = weighted_loss(parts, hp)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite optimization loss.")
        loss.backward()
        optimizer.step()
        n = target.numel()
        count += n
        totals["total_loss"] += loss.item() * n
        for name, value in parts.items():
            totals[name] += value.item() * n
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
@preserve_random_state()
def evaluate_training_pairs(model: PairwiseCNNVAE, data, entries: list[dict],
                            hp: Hyperparameters, batch_size: int,
                            device: torch.device, seed: int) -> tuple[dict, list[dict]]:
    """One seeded posterior sample per side; reuse it for the reverse order."""
    seed_everything(seed)
    model.eval()
    loader = make_loader(data, entries, batch_size, ordered=False, shuffle=False, seed=seed)
    totals = {name: 0.0 for name in (
        "total_loss", "reconstruction_bce", "kl_divergence", "ranking_bce")}
    targets, predictions, rows = [], [], []
    tile_mismatches, total_tiles = 0, 0
    offset = 0
    for x_a, x_b, target in loader:
        x_a, x_b, target = x_a.to(device), x_b.to(device), target.to(device)
        output = model(x_a, x_b)
        reverse_logits = model.preference_logits(output["z_b"], output["z_a"])
        parts = loss_components(output, x_a, x_b, target, hp.background_error_weight)
        parts["ranking_bce"] = 0.5 * (
            parts["ranking_bce"] + F.binary_cross_entropy_with_logits(reverse_logits, 1 - target)
        )
        parts["total_loss"] = weighted_loss(parts, hp)
        n = target.numel()
        for name, value in parts.items():
            if not torch.isfinite(value):
                raise FloatingPointError(f"Non-finite reporting loss: {name}.")
            totals[name] += value.item() * n
        for original, reconstruction in (
            (x_a, output["reconstruction_logits_a"]),
            (x_b, output["reconstruction_logits_b"]),
        ):
            input_tiles = original[:, :21].argmax(dim=1)
            decoded_tiles = reconstruction[:, :21].argmax(dim=1)
            tile_mismatches += (input_tiles != decoded_tiles).sum().item()
            total_tiles += input_tiles.numel()
        forward_probs = torch.sigmoid(output["rank_logits"]).cpu().tolist()
        reverse_probs = torch.sigmoid(reverse_logits).cpu().tolist()
        for index, (p_forward, p_reverse) in enumerate(zip(forward_probs, reverse_probs, strict=True)):
            entry = entries[offset + index]
            winner, loser = entry["preferred_level_id"], entry["less_preferred_level_id"]
            for first, second, label, probability in (
                (winner, loser, 1, p_forward), (loser, winner, 0, p_reverse)
            ):
                predicted = int(probability >= 0.5)
                targets.append(label)
                predictions.append(predicted)
                rows.append({"instance_id": entry["instance_id"], "first_level": first,
                             "second_level": second, "target": label,
                             "probability_first_preferred": probability, "prediction": predicted})
        offset += n
    metrics = {name: value / len(entries) for name, value in totals.items()}
    metrics["normalized_hamming_distance"] = tile_mismatches / total_tiles
    metrics.update(classification_metrics(targets, predictions))
    metrics.update({"comparisons": len(entries), "reporting_seed": seed,
                    "latent_policy": "one sampled latent per playthrough occurrence; reused in both orders",
                    "dropout_enabled": False})
    return metrics, rows


def split_training_groups(entries: list[dict], fraction: float, seed: int) -> tuple[list, list]:
    groups = {}
    for entry in entries:
        groups.setdefault(entry["split_group_id"], []).append(entry)
    if len(groups) < 2:
        return entries, []
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    target = max(1, round(len(entries) * fraction))
    validation_ids, count = set(), 0
    for group_id in group_ids[:-1]:  # Always leave at least one whole fitting group.
        if count >= target:
            break
        validation_ids.add(group_id)
        count += len(groups[group_id])
    fitting = [e for e in entries if e["split_group_id"] not in validation_ids]
    validation = [e for e in entries if e["split_group_id"] in validation_ids]
    return fitting, validation


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def flatten_epoch_metrics(row: dict) -> dict:
    flat = {"epoch": row["epoch"], "seconds": row["seconds"]}
    for name in LOSS_NAMES:
        flat[f"optimization_{name}"] = row["optimization"][name]
    for portion in ("train", "validation"):
        metrics = row[portion]
        for name in (*LOSS_NAMES, "accuracy", "f1_score", "normalized_hamming_distance"):
            flat[f"{portion}_{name}"] = metrics[name] if metrics is not None else None
    return flat


def save_epoch_checkpoint(path: Path, model: PairwiseCNNVAE,
                          optimizer: torch.optim.Optimizer, config: dict,
                          data, fitting: list[dict], validation: list[dict],
                          row: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "epoch": row["epoch"],
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "channels": data.channels,
        "selected_instance_ids": [entry["instance_id"] for entry in data.entries],
        "fitting_instance_ids": [entry["instance_id"] for entry in fitting],
        "validation_instance_ids": [entry["instance_id"] for entry in validation],
        "train_metrics": row["train"],
        "validation_metrics": row["validation"],
    }, temporary)
    temporary.replace(path)


def run_combination(data, fitting: list[dict], validation: list[dict],
                    hp: Hyperparameters, args, device: torch.device,
                    output: Path, combination_id: str, run_config: dict) -> dict:
    output.mkdir()
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    config = {**run_config, "combination_id": combination_id,
              "hyperparameters": asdict(hp)}
    write_json(output / "config.json", config)
    seed_everything(args.seed)
    model = PairwiseCNNVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    loader = make_loader(data, fitting, args.batch_size, ordered=True,
                         shuffle=True, seed=args.seed)
    history, csv_rows, saved_checkpoints, checkpoint_previews = [], [], [], []
    print(f"Starting {combination_id} with {asdict(hp)}", flush=True)
    try:
        for epoch in range(1, args.epochs + 1):
            start = time.monotonic()
            optimization = train_one_epoch(model, loader, optimizer, hp, device)
            train_metrics, train_predictions = evaluate_training_pairs(
                model, data, fitting, hp, args.batch_size, device, config["reporting_seed"]
            )
            validation_metrics, validation_predictions = None, []
            if validation:
                validation_metrics, validation_predictions = evaluate_training_pairs(
                    model, data, validation, hp, args.batch_size, device, config["reporting_seed"]
                )
            row = {"epoch": epoch, "optimization": optimization,
                   "train": train_metrics, "validation": validation_metrics,
                   "seconds": time.monotonic() - start}
            history.append(row)
            csv_rows.append(flatten_epoch_metrics(row))
            write_json(output / "train_history.json", history)
            write_csv(output / "epoch_metrics.csv", csv_rows)
            for portion, metrics in (("train", train_metrics), ("validation", validation_metrics)):
                if metrics is not None:
                    print(f"{combination_id} epoch {epoch}/{args.epochs} {portion} "
                          f"total={metrics['total_loss']:.6f} "
                          f"rec={metrics['reconstruction_bce']:.6f} "
                          f"KL={metrics['kl_divergence']:.6f} "
                          f"rank={metrics['ranking_bce']:.6f} "
                          f"accuracy={metrics['accuracy']:.4f} "
                          f"F1={metrics['f1_score']:.4f} "
                          f"Hamming={metrics['normalized_hamming_distance']:.6f}", flush=True)
            if checkpoint_epoch_selected(epoch, args.checkpoint_ranges):
                checkpoint = checkpoints / f"epoch_{epoch:03d}.pt"
                save_epoch_checkpoint(checkpoint, model, optimizer, config, data,
                                      fitting, validation, row)
                saved_checkpoints.append(checkpoint.relative_to(output).as_posix())
                preview_directory = output / "reconstructions" / checkpoint.stem
                with preserve_random_state():
                    seed_everything(config["preview_seed"])
                    preview = save_checkpoint_reconstructions(
                        model, data, config["validation_preview_samples"], preview_directory,
                        device, config["preview_seed"], checkpoint.relative_to(output).as_posix(), epoch,
                        selection_seed=config["seed"],
                    )
                relative_directory = preview_directory.relative_to(output).as_posix()
                checkpoint_previews.append({
                    "epoch": epoch,
                    "checkpoint": checkpoint.relative_to(output).as_posix(),
                    "sample_count": preview["sample_count"],
                    "comparison_image": (f"{relative_directory}/{preview['comparison_image']}"
                                         if preview["comparison_image"] is not None else None),
                    "samples_file": f"{relative_directory}/samples.json",
                })
                write_json(output / "checkpoint_previews.json", checkpoint_previews)
                if preview["sample_count"]:
                    print(f"{combination_id} epoch {epoch} validation reconstructions: "
                          f"{preview_directory / preview['comparison_image']}", flush=True)
    finally:
        if history:
            plot_loss_curves(history, asdict(hp), output / "validation_curves.png", combination_id)
    write_json(output / "train_metrics.json", train_metrics)
    write_json(output / "validation_metrics.json", validation_metrics)
    write_csv(output / "train_predictions.csv", train_predictions)
    if validation_predictions:
        write_csv(output / "validation_predictions.csv", validation_predictions)
    return {
        "combination_id": combination_id,
        "directory": output.name,
        "hyperparameters": asdict(hp),
        "status": "completed",
        "completed_epochs": len(history),
        "last_train_metrics": train_metrics,
        "last_validation_metrics": validation_metrics,
        "checkpoint_files": saved_checkpoints,
        "checkpoint_previews": checkpoint_previews,
        "loss_graph": f"{output.name}/validation_curves.png",
    }


def write_report(output: Path, config: dict, experiments: list[dict]) -> None:
    architecture = config["architecture"]
    lines = ["# Hyperparameter comparison", "",
             f"Architecture: `{architecture['architecture_version']}`; "
             f"input shape: `{tuple(architecture['input_shape'])}` (channel, row, column).",
             f"Preprocessing drops the first {architecture['dropped_leading_columns']} columns "
             f"of each {architecture['source_width']}-column source level and its path mask.",
             f"Each configuration trains for {config['epochs']} epochs.",
             f"The fixed group split contains {config['fitting_comparisons']} fitting comparisons "
             f"and {config['validation_comparisons']} validation comparisons.",
             ("Training metrics use train_set; validation metrics use the saved validation_set."
              if config.get("validation_source") == "validation_set" else
              "Training metrics refer to the fitting portion. Both portions come from train_set."),
             "Choose a configuration and epoch from its curves and per-epoch metrics.",
             "Tile reconstruction errors are multiplied by background_error_weight only when the",
             "actual tile is outside air/ground and the tile argmax predicts air (-) or ground (X/#).",
             "The gate is a hard argmax; path, KL, and ranking losses retain their usual weights.",
             "Losses use fixed cell/bit denominators; background_error_weight=1 recovers the base objective.",
             "", "| Configuration | Learning rate | Rec weight | KL weight | Rank weight | Background error weight | Epochs | Final train accuracy | Final validation accuracy | Checkpoints | Validation curves |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for experiment in experiments:
        hp = experiment["hyperparameters"]
        validation = experiment["last_validation_metrics"]
        validation_accuracy = f"{validation['accuracy']:.6f}" if validation is not None else "unavailable"
        directory = experiment["directory"]
        lines.append(
            f"| {experiment['combination_id']} | {hp['learning_rate']} | {hp['lambda_rec']} | "
            f"{hp['lambda_kl']} | {hp['lambda_rank']} | {hp['background_error_weight']} | "
            f"{experiment['completed_epochs']} | "
            f"{experiment['last_train_metrics']['accuracy']:.6f} | {validation_accuracy} | "
            f"{len(experiment['checkpoint_files'])} | [Curves]({directory}/validation_curves.png) |"
        )
    saving_summary = checkpoint_summary(config["checkpoint_ranges"], config["epochs"])
    lines += ["", "Each graph contains six panels for total loss, reconstruction BCE, KL divergence,",
              "ranking BCE, ranking accuracy, and normalized tile Hamming distance.",
              "The panels show fitting and validation curves against num_epochs.",
              "Each configuration's train_history.json and epoch_metrics.csv record every epoch's losses,",
              "overall accuracy, binary F1, and normalized tile Hamming distance.",
              "Hamming distance compares each input tile grid with its own decoder reconstruction,",
              "using the highest-logit geometry tile at each position and averaging over both inputs.",
              "", saving_summary,
              "Each checkpoint filename identifies its saved epoch, such as epoch_021.pt for epoch 21."]
    previews = [(experiment, preview) for experiment in experiments
                for preview in experiment.get("checkpoint_previews", [])]
    if previews:
        lines += ["", "## Saved-model validation reconstructions", "",
                  f"Up to {PREVIEW_COUNT} distinct validation levels are randomly selected once and reused",
                  "for every checkpoint and configuration. Each sheet shows the original input followed",
                  "by its VAE reconstruction from a sampled posterior latent vector. Sampling uses the",
                  "same seed for each checkpoint. Black X markers show the input or reconstructed path.",
                  "Tile channels use argmax; the independent path channel uses probability >= 0.5."]
        for experiment, preview in previews:
            directory = experiment["directory"]
            lines += ["", f"### {experiment['combination_id']} / epoch {preview['epoch']}", "",
                      f"[Saved model]({directory}/{preview['checkpoint']}) | "
                      f"[Sample details]({directory}/{preview['samples_file']})", ""]
            if preview["comparison_image"] is not None:
                lines.append(f"![{preview['sample_count']} validation inputs and VAE reconstructions]"
                             f"({directory}/{preview['comparison_image']})")
            else:
                lines.append("No validation levels were available for reconstruction previews.")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Preprocessed root containing train_set, channels.json and manifest.json")
    parser.add_argument("--hyperparameters-file", type=Path, default=DEFAULT_HYPERPARAMETERS,
                        help="Explicit learning_rate, lambda_rec, lambda_kl, lambda_rank, background_error_weight rows; default hyperparameter-combinations.txt beside train.py")
    parser.add_argument("--num-training-samples", type=int, default=NUM_TRAINING_SAMPLES,
                        help="Cap on original comparisons, keeping whole groups; default all")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Complete training duration for every hyperparameter combination")
    add_checkpoint_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Ordered examples per optimization batch")
    parser.add_argument("--validation-fraction", type=float, default=0.2,
                        help="Fallback validation fraction for datasets without a saved validation_set")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        help="New or empty comparison directory, default a timestamped run")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    validate_checkpoint_ranges(parser, args, CHECKPOINT_START_EPOCH, CHECKPOINT_END_EPOCH)
    if args.num_training_samples is not None and args.num_training_samples < 1:
        parser.error("--num-training-samples must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between zero and one")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32 - 1]")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    combinations = load_hyperparameter_combinations(args.hyperparameters_file)
    torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)
    data = load_training_data(args.dataset.resolve(), args.num_training_samples, args.seed)
    if data.validation_entries:
        fitting, validation = data.entries, data.validation_entries
        validation_source = "validation_set"
    else:
        fitting, validation = split_training_groups(data.entries, args.validation_fraction, args.seed)
        validation_source = "train_set"
    preview_samples = select_validation_levels(validation, args.seed)
    output = (args.output_dir or HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be new or empty {output}")
    if output == args.dataset.resolve() or args.dataset.resolve() in output.parents:
        raise ValueError("Output directory must be outside the preprocessed dataset.")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "independent runs for all specified combinations, user selects configuration and epoch",
        "hyperparameters_file": str(args.hyperparameters_file.resolve()),
        "hyperparameter_combinations": [asdict(hp) for hp in combinations],
        "dataset": str(args.dataset.resolve()), "dataset_fingerprint": data.dataset_fingerprint,
        "requested_sample_cap": args.num_training_samples,
        "available_comparisons": data.available_comparisons,
        "selected_comparisons": len(data.entries),
        "fitting_comparisons": len(fitting), "validation_comparisons": len(validation),
        "epochs": args.epochs, "combination_count": len(combinations),
        "checkpoint_start_epoch": args.checkpoint_start_epoch,
        "checkpoint_end_epoch": args.checkpoint_end_epoch,
        "checkpoint_ranges": args.checkpoint_ranges,
        "batch_size": args.batch_size,
        "validation_source": validation_source,
        "validation_fraction": args.validation_fraction if validation_source == "train_set" else None,
        "seed": args.seed, "reporting_seed": (args.seed + 1) % (2**32),
        "preview_seed": (args.seed + 2) % (2**32),
        "validation_preview_requested_levels": PREVIEW_COUNT,
        "validation_preview_samples": preview_samples,
        "device": str(device), "cpu_threads": args.cpu_threads,
        "architecture": architecture_metadata(),
        "python_version": platform.python_version(), "pytorch_version": str(torch.__version__),
        "numpy_version": np.__version__, "platform": platform.platform(),
        "parameter_count": sum(p.numel() for p in PairwiseCNNVAE().parameters()),
    }
    write_json(output / "run_config.json", config)
    write_json(output / "channels.json", data.channels)
    write_json(output / "selected_train_comparisons.json", data.entries)
    if data.validation_entries:
        write_json(output / "selected_validation_comparisons.json", data.validation_entries)
    write_json(output / "split.json", {
        "fitting_instance_ids": [entry["instance_id"] for entry in fitting],
        "validation_instance_ids": [entry["instance_id"] for entry in validation],
        "policy": ("saved train_set/validation_set assignments reused for all configurations"
                   if validation_source == "validation_set" else
                   "one fixed split of whole split_group_id groups for all configurations"),
    })
    experiments = [
        {"combination_id": f"combination_{index:02d}",
         "directory": hyperparameter_directory_name(hp),
         "hyperparameters": asdict(hp), "status": "pending"}
        for index, hp in enumerate(combinations, 1)
    ]
    write_json(output / "experiments.json", experiments)
    print(f"Loaded {len(data.entries)} comparisons, fitting={len(fitting)}, "
          f"validation={len(validation)}, configurations={len(experiments)}, device={device}", flush=True)
    for index, hp in enumerate(combinations):
        combination_id = experiments[index]["combination_id"]
        experiments[index]["status"] = "running"
        write_json(output / "experiments.json", experiments)
        try:
            experiments[index] = run_combination(
                data, fitting, validation, hp, args, device,
                output / experiments[index]["directory"], combination_id, config
            )
        except BaseException as error:
            experiments[index]["status"] = "failed"
            experiments[index]["error"] = str(error)
            write_json(output / "experiments.json", experiments)
            raise
        write_json(output / "experiments.json", experiments)
    write_report(output, config, experiments)
    print(f"Completed all {len(experiments)} configurations. Review {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
