"""Save training and validation curves for one hyperparameter combination."""

from pathlib import Path


def plot_loss_curves(
    history: list[dict],
    hyperparameters: dict,
    output_path: Path,
    combination_id: str,
) -> None:
    """Save six measurement panels as a headless PNG figure."""
    if not history:
        raise ValueError("Measurement history must contain at least one epoch.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    marker = "o" if len(epochs) == 1 else None
    last_epoch = epochs[-1]
    epoch_ticks = sorted({1 + round((last_epoch - 1) * index / 6) for index in range(7)})
    measurements = (
        ("total_loss", "Weighted total loss", "Loss"),
        ("reconstruction_bce", "Reconstruction BCE", "Loss"),
        ("kl_divergence", "KL divergence", "Loss"),
        ("ranking_bce", "Fun-ranking BCE", "Loss"),
        ("accuracy", "Ranking accuracy", "Accuracy"),
        (
            "normalized_hamming_distance",
            "Normalized tile Hamming distance",
            "Normalized Hamming distance",
        ),
    )
    has_validation = any(row["validation"] is not None for row in history)
    figure, axes = plt.subplots(3, 2, figsize=(12, 10.5), layout="constrained")
    try:
        parameter_text = "   ".join(
            f"{name}={hyperparameters[name]:g}"
            for name in ("learning_rate", "lambda_rec", "lambda_kl", "lambda_rank")
        )
        figure.suptitle(f"{combination_id}\n{parameter_text}", fontsize=12)
        for axis, (measurement_key, title, y_label) in zip(axes.flat, measurements, strict=True):
            axis.plot(
                epochs,
                [row["train"][measurement_key] for row in history],
                color="#2563eb",
                linewidth=1.8,
                linestyle="-",
                marker=marker,
                label="Train",
            )
            if has_validation:
                axis.plot(
                    epochs,
                    [
                        row["validation"][measurement_key]
                        if row["validation"] is not None
                        else float("nan")
                        for row in history
                    ],
                    color="#dc2626",
                    linewidth=1.8,
                    linestyle="--",
                    marker=marker,
                    label="Validation",
                )
            axis.set_title(title, fontsize=11)
            axis.set_xlabel("num_epochs")
            axis.set_ylabel(y_label)
            if measurement_key in ("accuracy", "normalized_hamming_distance"):
                axis.set_ylim(0, 1)
            axis.set_xticks(epoch_ticks)
            axis.set_xlim((0.8, 1.2) if last_epoch == 1 else (1, last_epoch))
            axis.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.65)
            axis.legend(frameon=False, fontsize=9)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="png", dpi=180)
    finally:
        plt.close(figure)
