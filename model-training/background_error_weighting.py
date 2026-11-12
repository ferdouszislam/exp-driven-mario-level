"""Increase tile reconstruction loss when a non-background tile becomes background."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


TILE_CHANNELS = 21
BACKGROUND_CHANNELS = (0, 2, 19)  # X (solid), - (empty), # (solid alias).


def background_error_weights(logits: Tensor, target: Tensor,
                             background_error_weight: float) -> Tensor:
    """Return BHW weights for non-background targets decoded as background.

    Only tile channels participate in the hard argmax decision. The path channel
    and mushroom stalk (|, channel 20) are outside the background group. Argmax
    supplies a fixed mask during backpropagation; gradients flow through the
    weighted reconstruction terms instead of through the decision itself.
    """
    weight = float(background_error_weight)
    if not math.isfinite(weight) or weight < 1:
        raise ValueError("background_error_weight must be finite and at least 1.")

    predicted_tiles = logits[:, :TILE_CHANNELS].argmax(dim=1)
    target_tiles = target[:, :TILE_CHANNELS].argmax(dim=1)
    predicted_background = torch.zeros_like(predicted_tiles, dtype=torch.bool)
    target_background = torch.zeros_like(target_tiles, dtype=torch.bool)
    for channel in BACKGROUND_CHANNELS:
        predicted_background |= predicted_tiles == channel
        target_background |= target_tiles == channel
    selected = predicted_background & ~target_background
    return torch.where(selected, logits.new_tensor(weight), logits.new_tensor(1.0))


def weighted_bce_reconstruction(logits: Tensor, target: Tensor,
                                background_error_weight: float) -> Tensor:
    """Average weighted tile BCE and unchanged path BCE over B*22*H*W.

    At selected cells all 21 tile BCE terms receive the multiplier. The fixed
    element-count denominator preserves the size of the extra penalty.
    """
    weights = background_error_weights(logits, target, background_error_weight)
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return torch.cat((
        losses[:, :TILE_CHANNELS] * weights.unsqueeze(1),
        losses[:, TILE_CHANNELS:TILE_CHANNELS + 1],
    ), dim=1).mean()


def weighted_categorical_reconstruction_loss_components(
        logits: Tensor, target: Tensor,
        background_error_weight: float) -> dict[str, Tensor]:
    """Return tile CE with fixed cell-count averaging and unchanged path BCE."""
    weights = background_error_weights(logits, target, background_error_weight)
    tile_losses = F.cross_entropy(
        logits[:, :TILE_CHANNELS], target[:, :TILE_CHANNELS].argmax(dim=1),
        reduction="none",
    )
    return {
        "tile_cross_entropy": (weights * tile_losses).mean(),
        "path_bce": F.binary_cross_entropy_with_logits(
            logits[:, TILE_CHANNELS], target[:, TILE_CHANNELS],
        ),
    }
