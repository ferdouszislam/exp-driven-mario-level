"""Rank epochs with both train and validation accuracy > 0.5 and plot metrics.

Run from the repository root:
    .venv/bin/python model-training/analyze_runs.py
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation, localcontext
import json
import math
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
MIN_ACCURACY = Decimal("0.5")
HYPERPARAMETERS = ("learning_rate", "lambda_rec", "lambda_kl", "lambda_rank")
METRICS = (
    "total_loss", "reconstruction_bce", "kl_divergence", "ranking_bce",
    "accuracy", "normalized_hamming_distance",
)
METRIC_COLUMNS = tuple(f"{split}_{metric}" for metric in METRICS
                       for split in ("train", "validation"))
RANKINGS = {
    "validation_accuracy": ("top_validation_accuracy.csv", "Highest validation accuracy"),
    "accuracy_gap": ("smallest_accuracy_gap.csv", "Smallest absolute accuracy gap"),
}


def decimal_value(value: str, location: str) -> Decimal:
    """Parse a finite logged number without introducing binary float rounding."""
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError(f"{location}: expected a numeric value, got {value!r}.") from error
    if not number.is_finite() or not math.isfinite(float(number)):
        raise ValueError(f"{location}: expected a finite value, got {value!r}.")
    return number


def load_experiment(experiment_dir: Path) -> list[dict]:
    """Read every saved epoch, retaining original metric strings and config IDs."""
    records = []
    combination_ids = set()
    metric_files = sorted(experiment_dir.glob("*/epoch_metrics.csv"))
    if not metric_files:
        raise ValueError(f"{experiment_dir}: no combination epoch_metrics.csv files found.")
    for metrics_path in metric_files:
        config_path = metrics_path.parent / "config.json"
        if not config_path.is_file():
            raise ValueError(f"{config_path}: missing configuration for saved metrics.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        combination_id = config.get("combination_id", "")
        match = re.fullmatch(r"combination_(\d+)", combination_id)
        if not match:
            raise ValueError(f"{config_path}: invalid combination_id {combination_id!r}.")
        combination_number = int(match.group(1))
        if combination_number < 1 or combination_number in combination_ids:
            raise ValueError(f"{config_path}: invalid or duplicate combination number.")
        combination_ids.add(combination_number)
        hyperparameters = config.get("hyperparameters", {})
        for name in HYPERPARAMETERS:
            if name not in hyperparameters:
                raise ValueError(f"{config_path}: missing hyperparameter {name}.")
            decimal_value(str(hyperparameters[name]), f"{config_path}: {name}")
        architecture = config.get("architecture", {})
        metadata = {
            "experiment_version": experiment_dir.name,
            "model_class": architecture.get("model_class", ""),
            "architecture_version": architecture.get("architecture_version", ""),
            "combination_number": combination_number,
            "combination_id": combination_id,
            "combination_directory": metrics_path.parent.name,
            **{name: hyperparameters[name] for name in HYPERPARAMETERS},
            "batch_size": config.get("batch_size", ""),
            "latent_dim": architecture.get("latent_dim", ""),
            "dropout": architecture.get("dropout", ""),
            "seed": config.get("seed", ""),
            "reporting_seed": config.get("reporting_seed", ""),
            "source_metrics_csv": metrics_path.relative_to(experiment_dir.parent).as_posix(),
            "source_config_json": config_path.relative_to(experiment_dir.parent).as_posix(),
        }
        seen_epochs = set()
        with metrics_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = {"epoch", *METRIC_COLUMNS} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{metrics_path}: missing columns: {', '.join(sorted(missing))}.")
            for line_number, row in enumerate(reader, 2):
                location = f"{metrics_path}:{line_number}"
                try:
                    epoch = int(row["epoch"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{location}: invalid epoch {row['epoch']!r}.") from error
                if epoch < 1 or epoch in seen_epochs:
                    raise ValueError(f"{location}: invalid or duplicate epoch {epoch}.")
                seen_epochs.add(epoch)
                for column in METRIC_COLUMNS:
                    value = decimal_value(row[column], f"{location}: {column}")
                    if column.endswith(("accuracy", "normalized_hamming_distance")) and not 0 <= value <= 1:
                        raise ValueError(f"{location}: {column} must be between 0 and 1.")
                # These strings have the full precision stored by the training script.
                with localcontext() as context:
                    context.prec = 50
                    gap = abs(Decimal(row["train_accuracy"]) - Decimal(row["validation_accuracy"]))
                record = {**metadata, "epoch": epoch, "accuracy_gap": gap,
                          **{column: row[column] for column in METRIC_COLUMNS}}
                # Keep the additional reported train/validation metric when present.
                for column in ("train_f1_score", "validation_f1_score"):
                    record[column] = row.get(column, "")
                records.append(record)
        if not seen_epochs:
            raise ValueError(f"{metrics_path}: no epoch rows found.")
    return records


def is_eligible(record: dict) -> bool:
    """Require both saved accuracies to be strictly above the threshold."""
    return all(Decimal(str(record[f"{split}_accuracy"])) > MIN_ACCURACY
               for split in ("train", "validation"))


def rank_records(records: list[dict], criterion: str, top_n: int = 10) -> list[dict]:
    """Rank eligible combination/epoch pairs; a combination may appear repeatedly."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if criterion not in RANKINGS:
        raise ValueError(f"Unknown ranking criterion: {criterion}.")

    def sort_key(record: dict) -> tuple:
        accuracy = Decimal(str(record["validation_accuracy"]))
        gap = record["accuracy_gap"]
        primary = (-accuracy, gap) if criterion == "validation_accuracy" else (gap, -accuracy)
        return (*primary, record["combination_number"], record["epoch"])

    eligible = [record for record in records if is_eligible(record)]
    return sorted(eligible, key=sort_key)[:top_n]


def plot_record(record: dict, output_path: Path) -> None:
    """Save one PNG with six train/validation bar charts for this exact epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = (
        "Weighted total loss", "Reconstruction BCE", "KL divergence",
        "Fun-ranking BCE", "Ranking accuracy", "Normalized tile Hamming distance",
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout="constrained")
    try:
        parameter_text = "   ".join(f"{name}={record[name]:g}" for name in HYPERPARAMETERS)
        figure.suptitle(
            f"{record['experiment_version']}  |  {record['architecture_version']}\n"
            f"{record['combination_id']}  |  Epoch {record['epoch']}  |  "
            f"Absolute accuracy gap: {float(record['accuracy_gap']):.6g}\n{parameter_text}",
            fontsize=13,
        )
        for axis, metric, title in zip(axes.flat, METRICS, titles, strict=True):
            values = [float(record[f"{split}_{metric}"]) for split in ("train", "validation")]
            bars = axis.bar(["Train", "Validation"], values, width=0.58,
                            color=["#2563eb", "#dc2626"])
            axis.bar_label(bars, labels=[f"{value:.6g}" for value in values], padding=5, fontsize=10)
            axis.set_title(title, fontsize=12, pad=12)
            axis.set_ylabel("Loss" if metric not in ("accuracy", "normalized_hamming_distance")
                            else "Accuracy" if metric == "accuracy" else "Mismatch fraction")
            if metric in ("accuracy", "normalized_hamming_distance"):
                # The small amount of headroom leaves labels visible for a value of 1.
                axis.set_ylim(0, 1.12)
                axis.set_yticks([0, 0.25, 0.5, 0.75, 1])
            else:
                axis.set_ylim(min(0, min(values) * 1.2), max(max(values) * 1.22, 1e-12))
            axis.set_axisbelow(True)
            axis.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.7)
            axis.spines[["top", "right"]].set_visible(False)
        figure.supxlabel("Loss components are unweighted as logged; total loss already includes the lambda weights.",
                           fontsize=10)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="png", dpi=150)
    finally:
        plt.close(figure)


def write_csv(path: Path, records: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write ranking positions and full-precision source metrics."""
    fieldnames = ["rank", *(fieldnames if fieldnames is not None else records[0])]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, record in enumerate(records, 1):
            writer.writerow({"rank": rank, **record})
    temporary.replace(path)


def write_experiment_report(output: Path, records: list[dict], rankings: dict) -> None:
    """Add a compact, clickable index of the two rankings and their PNGs."""
    eligible_count = sum(is_eligible(record) for record in records)
    lines = [f"# {records[0]['experiment_version']}", "",
             f"Read {len(records):,} epoch records from "
             f"{len({record['combination_number'] for record in records})} combinations.", "",
             f"Ranked {eligible_count:,} eligible epochs with both train and validation accuracy strictly greater than 0.5.", "",
             "Each entry is a combination/epoch pair; combinations may repeat. "
             "The linked CSVs include hyperparameters and every requested train/validation metric.", "",
             "Accuracy gap = abs(train_accuracy - validation_accuracy). "
             "CSV values retain source precision; this preview and chart labels shorten the display."]
    for criterion, selected in rankings.items():
        filename, title = RANKINGS[criterion]
        lines += ["", f"## {title}", "", f"[Full results]({filename})", "",
                  "| Rank | Combination | Epoch | Train accuracy | Validation accuracy | Absolute gap | PNG |",
                  "|---:|---|---:|---:|---:|---:|---|"]
        if not selected:
            lines += ["", "No epochs have both train and validation accuracy > 0.5."]
        for rank, record in enumerate(selected, 1):
            lines.append(f"| {rank} | {record['combination_id']} | {record['epoch']} | "
                         f"{float(record['train_accuracy']):.6g} | "
                         f"{float(record['validation_accuracy']):.6g} | "
                         f"{float(record['accuracy_gap']):.6g} | [Chart]({record['plot_file']}) |")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=HERE / "runs",
                        help="Folder containing experiment versions (default: runs beside this script).")
    parser.add_argument("--output-dir", type=Path,
                        help="Report folder (default: run-analysis beside the selected runs folder).")
    parser.add_argument("--top-n", type=int, default=10, help="Results per ranking per version (default: 10).")
    parser.add_argument("--experiment", action="append", metavar="NAME",
                        help="Only this experiment folder; repeat to select more than one.")
    args = parser.parse_args(argv)
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_dir = args.runs_dir.resolve()
    output_dir = (args.output_dir or runs_dir.parent / "run-analysis").resolve()
    try:
        if not runs_dir.is_dir():
            raise ValueError(f"Runs directory does not exist: {runs_dir}")
        if output_dir == runs_dir or output_dir.is_relative_to(runs_dir):
            raise ValueError("--output-dir must be outside --runs-dir to keep source experiments separate.")
        if args.experiment:
            if any(Path(name).name != name or name in (".", "..") for name in args.experiment):
                raise ValueError("--experiment must be an immediate folder name inside --runs-dir.")
            directories = [runs_dir / name for name in sorted(set(args.experiment))]
        else:
            directories = [path for path in sorted(runs_dir.iterdir())
                           if path.is_dir() and any(path.glob("*/epoch_metrics.csv"))]
        if not directories:
            raise ValueError(f"No experiment versions containing epoch metrics found in {runs_dir}.")
        # Validate all inputs before producing reports.
        experiments = [(directory, load_experiment(directory)) for directory in directories]
        output_dir.mkdir(parents=True, exist_ok=True)
        index = ["# Saved epoch rankings", "", f"Source runs: `{runs_dir}`", "",
                 f"Up to {args.top_n} entries per ranking, separately for each experiment version.", "",
                 "1. Highest validation accuracy; ties use the smaller absolute train/validation accuracy gap.",
                 "2. Smallest absolute accuracy gap; ties use the higher validation accuracy.", "",
                 "Remaining ties use combination number, then epoch number (both ascending). "
                 "Only epochs with both train and validation accuracy strictly greater than 0.5 are eligible; "
                 "multiple epochs from the same combination are allowed. "
                 "Gaps are calculated from the saved decimal values without display rounding or a tie tolerance.", "",
                 "Each unique selected combination/epoch has one PNG shared by both rankings if needed.", "",
                 "Source paths in CSVs are relative to the source runs folder. "
                 "PNG paths are relative to each experiment's report folder.", "",
                 "| Experiment version | Combinations | Epoch records | Eligible epochs (both > 0.5) | Selected PNGs | Results |",
                 "|---|---:|---:|---:|---:|---|"]
        total_pngs = 0
        for directory, records in experiments:
            output = output_dir / directory.name
            rankings = {criterion: rank_records(records, criterion, args.top_n) for criterion in RANKINGS}
            selected_epochs = {}
            for selected in rankings.values():
                for record in selected:
                    record["plot_file"] = f"plots/{record['combination_id']}_epoch_{record['epoch']:03d}.png"
                    selected_epochs[(record["combination_number"], record["epoch"])] = record
            eligible_count = sum(is_eligible(record) for record in records)
            print(f"{directory.name}: {eligible_count:,}/{len(records):,} eligible epochs; "
                  f"generating {len(selected_epochs)} PNGs...", flush=True)
            for record in selected_epochs.values():
                plot_record(record, output / record["plot_file"])
            for criterion, selected in rankings.items():
                write_csv(output / RANKINGS[criterion][0], selected,
                          fieldnames=list(records[0] | {"plot_file": ""}))
            write_experiment_report(output, records, rankings)
            # Remove only this script's obsolete chart files after the reports succeed.
            current_plots = {record["plot_file"] for record in selected_epochs.values()}
            for path in (output / "plots").glob("*.png"):
                if (re.fullmatch(r"combination_\d+_epoch_\d+\.png", path.name)
                        and path.relative_to(output).as_posix() not in current_plots):
                    path.unlink()
            combination_count = len({record["combination_number"] for record in records})
            index.append(f"| {directory.name} | {combination_count} | {len(records):,} | "
                         f"{eligible_count:,} | {len(selected_epochs)} | [Open]({directory.name}/README.md) |")
            total_pngs += len(selected_epochs)
        (output_dir / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
        print(f"Saved {len(experiments) * 2} ranking CSVs and {total_pngs} PNGs to {output_dir}")
    except (OSError, ValueError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
