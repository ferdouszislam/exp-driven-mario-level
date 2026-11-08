"""Shared CNN-VAE and ordered-pair FC tail from initial_model_architecture.md."""

from __future__ import annotations

import torch
from torch import Tensor, nn


INPUT_SHAPE = (22, 16, 192)
LATENT_DIM = 32
GROUP_NORM_GROUPS = 8


class PairwiseCNNVAE(nn.Module):
    """Encode/decode two playthroughs with shared weights and predict P(A > B).

    Reconstruction and preference outputs are logits, for stable BCE-with-logits
    losses. ``forward`` always uses reparameterized samples; callers choosing a
    deterministic reporting convention can explicitly pass encoder means to
    ``decode`` and ``preference_logits`` instead.
    """

    def __init__(self) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        for in_channels, out_channels in ((22, 32), (32, 64), (64, 128), (128, 128)):
            encoder_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.GroupNorm(GROUP_NORM_GROUPS, out_channels),
                    nn.LeakyReLU(negative_slope=0.2),
                ]
            )
        self.encoder_convs = nn.Sequential(*encoder_layers)
        self.encoder_hidden = nn.Sequential(
            nn.Flatten(), nn.Linear(1536, 128), nn.LeakyReLU(negative_slope=0.2)
        )
        self.mu_head = nn.Linear(128, LATENT_DIM)
        self.log_var_head = nn.Linear(128, LATENT_DIM)

        self.decoder_projection = nn.Linear(LATENT_DIM, 1536)
        decoder_layers: list[nn.Module] = []
        for in_channels, out_channels in ((128, 128), (128, 64), (64, 32)):
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(
                        in_channels, out_channels, kernel_size=4, stride=2, padding=1
                    ),
                    nn.GroupNorm(GROUP_NORM_GROUPS, out_channels),
                    nn.LeakyReLU(negative_slope=0.2),
                ]
            )
        decoder_layers.append(nn.ConvTranspose2d(32, 22, kernel_size=4, stride=2, padding=1))
        self.decoder_convs = nn.Sequential(*decoder_layers)

        self.fc_tail = nn.Sequential(
            nn.Linear(2 * LATENT_DIM, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 1),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return posterior mean and log variance for one batch of playthroughs."""
        if x.ndim != 4 or tuple(x.shape[1:]) != INPUT_SHAPE:
            raise ValueError(f"Expected input shape (batch, {INPUT_SHAPE}), got {tuple(x.shape)}")
        hidden = self.encoder_hidden(self.encoder_convs(x))
        return self.mu_head(hidden), self.log_var_head(hidden)

    @staticmethod
    def reparameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        """Sample z = mu + exp(0.5 * log_var) * epsilon, epsilon ~ N(0, I)."""
        return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

    def decode(self, z: Tensor) -> Tensor:
        """Return 22-channel reconstruction logits from a latent batch."""
        projected = self.decoder_projection(z).reshape(-1, 128, 1, 12)
        return self.decoder_convs(projected)

    def preference_logits(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """Return one logit per ordered pair, with A concatenated before B."""
        return self.fc_tail(torch.cat((z_a, z_b), dim=1)).squeeze(-1)

    def forward(self, x_a: Tensor, x_b: Tensor) -> dict[str, Tensor]:
        """Sample both posteriors, reconstruct both inputs, and classify the pair."""
        mu_a, log_var_a = self.encode(x_a)
        mu_b, log_var_b = self.encode(x_b)
        z_a = self.reparameterize(mu_a, log_var_a)
        z_b = self.reparameterize(mu_b, log_var_b)
        return {
            "reconstruction_logits_a": self.decode(z_a),
            "reconstruction_logits_b": self.decode(z_b),
            "mu_a": mu_a,
            "mu_b": mu_b,
            "log_var_a": log_var_a,
            "log_var_b": log_var_b,
            "z_a": z_a,
            "z_b": z_b,
            "rank_logits": self.preference_logits(z_a, z_b),
        }


def architecture_metadata() -> dict[str, object]:
    """Return the fixed architecture choices to save beside a checkpoint."""
    return {
        "model_class": "PairwiseCNNVAE",
        "input_shape": list(INPUT_SHAPE),
        "latent_dim": LATENT_DIM,
        "group_norm_groups": GROUP_NORM_GROUPS,
        "negative_slope": 0.2,
        "dropout": 0.1,
        "encoder_channels": [22, 32, 64, 128, 128],
        "encoder_hidden": [1536, 128],
        "decoder_channels": [128, 128, 64, 32, 22],
        "fc_tail_sizes": [64, 64, 32, 1],
        "outputs": "reconstruction and preference logits",
    }
