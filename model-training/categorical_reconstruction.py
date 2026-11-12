"""Categorical tiles and an independent binary player-path reconstruction."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


TILE_CHANNELS = 21


def reconstruction_probabilities(logits: Tensor) -> Tensor:
    """Normalize tile types per cell; apply sigmoid only to the path logit."""
    return torch.cat((
        torch.softmax(logits[:, :TILE_CHANNELS], dim=1),
        torch.sigmoid(logits[:, TILE_CHANNELS:TILE_CHANNELS + 1]),
    ), dim=1)


def reconstruction_loss_components(logits: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Return mean tile CE and mean path BCE over batch and spatial positions.

    Cross-entropy includes log-softmax and BCE-with-logits includes sigmoid.
    Both consume raw logits, avoiding a second activation or log of rounded
    probabilities. The validated dataset has exactly one tile bit per cell.
    """
    return {
        "tile_cross_entropy": F.cross_entropy(
            logits[:, :TILE_CHANNELS], target[:, :TILE_CHANNELS].argmax(dim=1),
        ),
        "path_bce": F.binary_cross_entropy_with_logits(
            logits[:, TILE_CHANNELS], target[:, TILE_CHANNELS],
        ),
    }
