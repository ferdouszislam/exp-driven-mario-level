"""Save training and validation loss curves for one hyperparameter combination."""

from pathlib import Path


def plot_loss_curves(
    history: list[dict],
    hyperparameters: dict,
    output_path: Path,
    combination_id: str,
) -> None:
    """Save four independently scaled loss panels as a headless PNG figure."""
    if not history:
        raise ValueError("Loss history must contain at least one epoch.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    marker = "o" if len(epochs) == 1 else None
    last_epoch = epochs[-1]
    epoch_ticks = sorted({1 + round((last_epoch - 1) * index / 6) for index in range(7)})
    losses = (
        ("total_loss", "Weighted total loss"),
        ("reconstruction_bce", "Reconstruction BCE"),
        ("kl_divergence", "KL divergence"),
        ("ranking_bce", "Fun-ranking BCE"),
    )
    has_validation = any(row["validation"] is not None for row in history)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    try:
        parameter_text = "   ".join(
            f"{name}={hyperparameters[name]:g}"
            for name in ("learning_rate", "lambda_rec", "lambda_kl", "lambda_rank")
        )
        figure.suptitle(f"{combination_id}\n{parameter_text}", fontsize=12)
        for axis, (loss_key, title) in zip(axes.flat, losses, strict=True):
            axis.plot(
                epochs,
                [row["train"][loss_key] for row in history],
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
                        row["validation"][loss_key]
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
            axis.set_ylabel("Loss")
            axis.set_xticks(epoch_ticks)
            axis.set_xlim((0.8, 1.2) if last_epoch == 1 else (1, last_epoch))
            axis.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.65)
            axis.legend(frameon=False, fontsize=9)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, format="png", dpi=180)
    finally:
        plt.close(figure)
