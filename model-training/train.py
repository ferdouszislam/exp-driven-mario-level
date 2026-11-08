"""Train the proposal's shared CNN-VAE and concatenated-latent FC-Tail."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import ComparisonDataset, load_training_data
from model import PairwiseCNNVAE, architecture_metadata


# Set this to a small positive number for a quick run; None uses all train pairs.
# Counts original comparison folders BEFORE generating both input orders.
# Whole comparison groups are retained, so the actual count can be below this cap.
NUM_TRAINING_SAMPLES: int | None = None
EPOCHS = 50
TUNING_EPOCHS = 5
BATCH_SIZE = 16
SEED = 42
HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE.parent / "datasets/fun-study-dataset-analysis/preprocessed_dataset"


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float
    lambda_rec: float
    lambda_kl: float
    lambda_rank: float


# A small, declared joint search, rather than weights chosen using test results.
TUNING_CANDIDATES = (
    Hyperparameters(1e-3, 1.0, 1e-3, 1.0),
    Hyperparameters(1e-3, 1.0, 1e-2, 1.0),
    Hyperparameters(3e-4, 2.0, 1e-3, 1.0),
    Hyperparameters(3e-4, 1.0, 1e-3, 2.0),
)
SELECTION_RULE = (
    "smallest absolute fitting/validation total-loss gap; ties use candidate order"
)


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


def loss_components(output: dict, x_a: torch.Tensor, x_b: torch.Tensor,
                    target: torch.Tensor) -> dict[str, torch.Tensor]:
    """BCE averaged over bits; KL summed over latent coordinates, mean over inputs."""
    reconstruction = 0.5 * (
        F.binary_cross_entropy_with_logits(output["reconstruction_logits_a"], x_a)
        + F.binary_cross_entropy_with_logits(output["reconstruction_logits_b"], x_b)
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
    f1 = []
    for label in (0, 1):
        tp = matrix[label][label]
        fp, fn = matrix[1 - label][label], matrix[label][1 - label]
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {
        "accuracy": (matrix[0][0] + matrix[1][1]) / len(targets),
        "macro_f1": sum(f1) / 2,
        "f1_by_class": {"0": f1[0], "1": f1[1]},
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


def fit(model: PairwiseCNNVAE, data, entries: list[dict], hp: Hyperparameters,
        epochs: int, batch_size: int, device: torch.device, seed: int,
        phase: str) -> tuple[torch.optim.Optimizer, list[dict]]:
    loader = make_loader(data, entries, batch_size, ordered=True, shuffle=True, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {name: 0.0 for name in (
            "total_loss", "reconstruction_bce", "kl_divergence", "ranking_bce")}
        count = 0
        start = time.monotonic()
        for x_a, x_b, target in loader:
            x_a, x_b, target = x_a.to(device), x_b.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x_a, x_b)
            parts = loss_components(output, x_a, x_b, target)
            loss = weighted_loss(parts, hp)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {phase}, epoch {epoch}.")
            loss.backward()
            optimizer.step()
            n = target.numel()
            count += n
            totals["total_loss"] += loss.item() * n
            for name, value in parts.items():
                totals[name] += value.item() * n
        row = {"epoch": epoch, **{k: v / count for k, v in totals.items()},
               "seconds": time.monotonic() - start}
        history.append(row)
        print(f"{phase} epoch {epoch}/{epochs}: "
              f"total={row['total_loss']:.6f} rec={row['reconstruction_bce']:.6f} "
              f"KL={row['kl_divergence']:.6f} rank={row['ranking_bce']:.6f} "
              f"({row['seconds']:.2f}s)", flush=True)
    return optimizer, history


@torch.no_grad()
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
    offset = 0
    for x_a, x_b, target in loader:
        x_a, x_b, target = x_a.to(device), x_b.to(device), target.to(device)
        output = model(x_a, x_b)
        reverse_logits = model.preference_logits(output["z_b"], output["z_a"])
        parts = loss_components(output, x_a, x_b, target)
        parts["ranking_bce"] = 0.5 * (
            parts["ranking_bce"] + F.binary_cross_entropy_with_logits(reverse_logits, 1 - target)
        )
        parts["total_loss"] = weighted_loss(parts, hp)
        n = target.numel()
        for name, value in parts.items():
            if not torch.isfinite(value):
                raise FloatingPointError(f"Non-finite reporting loss: {name}.")
            totals[name] += value.item() * n
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


def tune(data, args, device: torch.device) -> tuple[Hyperparameters, dict]:
    fitting, validation = split_training_groups(data.entries, args.validation_fraction, args.seed)
    record = {
        "fitting_instance_ids": [e["instance_id"] for e in fitting],
        "validation_instance_ids": [e["instance_id"] for e in validation],
        "selection_rule": SELECTION_RULE,
        "selection_loss": "candidate-weighted total_loss on both portions",
        "comparison_policy": "post-fit evaluation mode; no gradients; seed + 1 on both portions",
        "selected_absolute_total_loss_gap": None,
        "candidates": [],
    }
    if not validation:
        record.update({"status": "skipped_single_group_smoke_run",
                       "reason": "At least two training comparison groups are needed for validation.",
                       "selected_hyperparameters": asdict(TUNING_CANDIDATES[0])})
        print("Only one selected group: provisional first candidate, no validation tuning.", flush=True)
        return TUNING_CANDIDATES[0], record
    for index, hp in enumerate(TUNING_CANDIDATES):
        seed_everything(args.seed)  # Same initialization/shuffle seed for all candidates.
        model = PairwiseCNNVAE().to(device)
        _, history = fit(model, data, fitting, hp, args.tuning_epochs, args.batch_size,
                         device, args.seed, f"tuning {index + 1}/{len(TUNING_CANDIDATES)}")
        # Compare the SAME fixed candidate on both portions, with dropout off.
        # Epoch history averages changing weights and active dropout, so it is
        # not the fitting loss used for this comparison.
        fitting_metrics, _ = evaluate_training_pairs(model, data, fitting, hp, args.batch_size,
                                                     device, args.seed + 1)
        validation_metrics, _ = evaluate_training_pairs(model, data, validation, hp, args.batch_size,
                                                        device, args.seed + 1)
        signed_gap = validation_metrics["total_loss"] - fitting_metrics["total_loss"]
        absolute_gap = abs(signed_gap)
        record["candidates"].append({"hyperparameters": asdict(hp),
                                     "fit_history": history, "fitting_metrics": fitting_metrics,
                                     "validation_metrics": validation_metrics,
                                     "validation_minus_fitting_total_loss": signed_gap,
                                     "absolute_total_loss_gap": absolute_gap})
        print(f"tuning {index + 1}: fitting total={fitting_metrics['total_loss']:.6f} "
              f"validation total={validation_metrics['total_loss']:.6f} "
              f"absolute gap={absolute_gap:.6f}", flush=True)
        del model
    # min retains the first candidate on an exact tie. Do not use the signed
    # difference: validation can be lower than fitting without having a small gap.
    selected = min(range(len(TUNING_CANDIDATES)),
                   key=lambda i: record["candidates"][i]["absolute_total_loss_gap"])
    record.update({"status": "completed", "selected_candidate_index": selected,
                   "selected_absolute_total_loss_gap": record["candidates"][selected]["absolute_total_loss_gap"],
                   "selected_hyperparameters": asdict(TUNING_CANDIDATES[selected])})
    return TUNING_CANDIDATES[selected], record


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_report(output: Path, config: dict, metrics: dict, tuning: dict) -> None:
    hp = config["hyperparameters"]
    lines = ["# Training results", "", "This run uses only the selected training comparisons.", "",
             f"- Comparisons: {metrics['comparisons']} of {config['available_comparisons']}",
             f"- Ordered examples: {metrics['ordered_examples']} (both orders, balanced labels)",
             f"- Final fitting epochs: {config['epochs']}", f"- Device: {config['device']}",
             f"- Hyperparameter selection: {tuning['status']}",
             f"- Hyperparameter selection rule: {tuning['selection_rule']}",
             f"- Adam learning rate: {hp['learning_rate']}",
             f"- Weights (reconstruction, KL, ranking): {hp['lambda_rec']}, {hp['lambda_kl']}, {hp['lambda_rank']}",
             ""]
    if tuning["status"] == "completed":
        candidate = tuning["candidates"][tuning["selected_candidate_index"]]
        lines += ["The selected tuning candidate was measured in evaluation mode on both portions:",
                  "", "| Tuning comparison | Value |", "|---|---:|",
                  f"| Fitting total loss | {candidate['fitting_metrics']['total_loss']:.10f} |",
                  f"| Validation total loss | {candidate['validation_metrics']['total_loss']:.10f} |",
                  f"| Validation minus fitting total loss | {candidate['validation_minus_fitting_total_loss']:.10f} |",
                  f"| Absolute total-loss gap (selection score) | {candidate['absolute_total_loss_gap']:.10f} |",
                  "", "This gap belongs to the tuning candidate before fresh final training.",
                  "A small gap can also occur when both losses are high; it does not guarantee",
                  "that the model avoids overfitting.", ""]
    else:
        lines += ["The tuning gap is unavailable because the selected data has only one group.", ""]
    lines += ["| Final train measurement | Value |", "|---|---:|"]
    for label, key in (("Weighted total loss", "total_loss"),
                       ("Reconstruction BCE (mean per bit)", "reconstruction_bce"),
                       ("KL divergence (sum per latent, mean per input)", "kl_divergence"),
                       ("Fun-ranking BCE", "ranking_bce"),
                       ("Fun-ranking accuracy", "accuracy"), ("Fun-ranking macro F1", "macro_f1")):
        lines.append(f"| {label} | {metrics[key]:.10f} |")
    lines += ["", "Final measurements use evaluation mode and one seeded posterior draw per input;",
              "the same two latent samples are reused for the two orders. These are in-sample",
              "training results, not estimates of held-out performance. No test examples are loaded.",
              "", f"Confusion matrix (true rows, predicted columns; labels 0, 1): `{metrics['confusion_matrix']}`.",
              "", "`model.pt` contains model and Adam states, architecture, codebook, selected IDs,",
              "configuration, dataset fingerprint and final train metrics. `train_predictions.csv`",
              "contains every ordered prediction. See `tuning.json` and `train_history.json` for",
              "selection details and stochastic optimization losses."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Preprocessed root containing train_set, channels.json and manifest.json")
    parser.add_argument("--num-training-samples", type=int, default=NUM_TRAINING_SAMPLES,
                        help="Cap on original comparisons, keeping whole groups; default: all")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--tuning-epochs", type=int, default=TUNING_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Ordered examples per optimization batch")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path,
                        help="New or empty output directory; defaults to a timestamped run")
    args = parser.parse_args(argv)
    for name in ("epochs", "tuning_epochs", "batch_size", "cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_training_samples is not None and args.num_training_samples < 1:
        parser.error("--num-training-samples must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between zero and one")
    if not 0 <= args.seed < 2**32 - 2:
        parser.error("--seed must be in [0, 2**32 - 3]")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    started = time.monotonic()
    torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)
    data = load_training_data(args.dataset.resolve(), args.num_training_samples, args.seed)
    output = (args.output_dir or HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output}")
    # Prevent output artifacts from altering a supplied dataset.
    if output == args.dataset.resolve() or args.dataset.resolve() in output.parents:
        raise ValueError("Output directory must be outside the preprocessed dataset.")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(data.entries)}/{data.available_comparisons} train comparisons "
          f"({2 * len(data.entries)} ordered examples); device={device}", flush=True)
    hp, tuning_record = tune(data, args, device)
    write_json(output / "tuning.json", tuning_record)
    print(f"Selected {asdict(hp)}; retraining from scratch on all selected train pairs.", flush=True)
    seed_everything(args.seed)
    model = PairwiseCNNVAE().to(device)
    optimizer, history = fit(model, data, data.entries, hp, args.epochs, args.batch_size,
                             device, args.seed, "final train")
    metrics, predictions = evaluate_training_pairs(model, data, data.entries, hp,
                                                   args.batch_size, device, args.seed + 2)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(args.dataset.resolve()),
        "dataset_fingerprint": data.dataset_fingerprint,
        "requested_sample_cap": args.num_training_samples,
        "available_comparisons": data.available_comparisons, "selected_comparisons": len(data.entries),
        "epochs": args.epochs, "tuning_epochs": args.tuning_epochs, "batch_size": args.batch_size,
        "validation_fraction": args.validation_fraction, "seed": args.seed,
        "device": str(device), "cpu_threads": args.cpu_threads,
        "hyperparameters": asdict(hp), "architecture": architecture_metadata(),
        "hyperparameter_selection_rule": tuning_record["selection_rule"],
        "selected_tuning_total_loss_gap": tuning_record.get("selected_absolute_total_loss_gap"),
        "python_version": platform.python_version(), "pytorch_version": str(torch.__version__),
        "numpy_version": np.__version__, "platform": platform.platform(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.monotonic() - started,
    }
    selected_ids = [e["instance_id"] for e in data.entries]
    write_json(output / "config.json", config)
    write_json(output / "selected_train_comparisons.json", data.entries)
    write_json(output / "train_history.json", history)
    write_json(output / "train_metrics.json", metrics)
    with (output / "train_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    torch.save({"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(), "config": config,
                "channels": data.channels, "selected_instance_ids": selected_ids,
                "train_metrics": metrics}, output / "model.pt")
    write_report(output, config, metrics, tuning_record)
    print(json.dumps({"output_dir": str(output), "train_metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
