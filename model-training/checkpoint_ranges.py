"""Shared checkpoint range arguments, epoch selection, and reporting."""

from __future__ import annotations

import argparse


def add_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint-start-epoch", type=int, nargs="+", action="extend",
        help="Inclusive start epochs; repeat or list values, paired with end epochs in order",
    )
    parser.add_argument(
        "--checkpoint-end-epoch", type=int, nargs="+", action="extend",
        help="Inclusive end epochs; supply one per start epoch; omit both to disable checkpoints",
    )


def validate_checkpoint_ranges(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    start_default: int | list[int] | None = None,
    end_default: int | list[int] | None = None,
) -> None:
    """Resolve local defaults without appending them to explicit CLI bounds."""
    for name, default in (("checkpoint_start_epoch", start_default),
                          ("checkpoint_end_epoch", end_default)):
        if getattr(args, name) is None and default is not None:
            setattr(args, name, [default] if isinstance(default, int) else list(default))

    starts, ends = args.checkpoint_start_epoch, args.checkpoint_end_epoch
    if (starts is None) != (ends is None):
        parser.error("Specify both --checkpoint-start-epoch and --checkpoint-end-epoch to save checkpoints")
    args.checkpoint_ranges = []
    if starts is None:
        return
    if not starts or len(starts) != len(ends):
        parser.error("Supply the same nonzero number of --checkpoint-start-epoch and --checkpoint-end-epoch values")
    for name, bounds in (("checkpoint_start_epoch", starts), ("checkpoint_end_epoch", ends)):
        if any(type(epoch) is not int or epoch < 1 for epoch in bounds):
            parser.error(f"--{name.replace('_', '-')} values must be positive integers")
    for index, (start, end) in enumerate(zip(starts, ends, strict=True), 1):
        if end < start:
            parser.error(f"--checkpoint-end-epoch must be at least --checkpoint-start-epoch in pair {index}")
        args.checkpoint_ranges.append((start, end))


def checkpoint_epoch_selected(epoch: int, ranges: list[tuple[int, int]]) -> bool:
    """Save an epoch once when it belongs to any inclusive range."""
    return any(start <= epoch <= end for start, end in ranges)


def checkpoint_summary(ranges: list[tuple[int, int]], epochs: int) -> str:
    """Describe the union of requested ranges within the training budget."""
    if not ranges:
        return "Checkpoint saving was disabled for this run."
    completed_ranges = sorted((start, min(end, epochs))
                              for start, end in ranges if start <= epochs)
    if not completed_ranges:
        return (f"Checkpoint saving begins at epoch {min(start for start, _ in ranges)}, "
                f"beyond this run's {epochs}-epoch budget.")
    merged = []
    for start, end in completed_ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    covered = ", ".join(str(start) if start == end else f"{start}–{end}"
                        for start, end in merged)
    return f"Checkpoints cover epochs {covered}, inclusive."
