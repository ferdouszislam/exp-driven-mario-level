"""Shared CNN-VAE and ordered-pair FC tail from initial_model_architecture.md."""

from __future__ import annotations

import torch
from torch import Tensor, nn


SOURCE_WIDTH = 192
DROPPED_LEADING_COLUMNS = 16
INPUT_SHAPE = (22, 16, SOURCE_WIDTH - DROPPED_LEADING_COLUMNS)
LATENT_DIM = 32
ARCHITECTURE_VERSION = "initial_model_architecture_crop16"
ENCODED_SHAPE = (128, 1, 11)
ENCODED_FEATURES = 1408
GROUP_NORM_GROUPS = 8
LEAKY_RELU_NEGATIVE_SLOPE = 0.2
FC_TAIL_DROPOUT = 0.1


class PairwiseCNNVAE(nn.Module):
    """Encode/decode two playthroughs with shared weights and predict P(A > B).

    Reconstruction and preference outputs are logits, for stable BCE-with-logits
    losses. ``forward`` always uses reparameterized samples; callers choosing a
    deterministic reporting convention can explicitly pass encoder means to
    ``decode`` and ``preference_logits`` instead.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder_convs = nn.Sequential(
            nn.Conv2d(22, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(GROUP_NORM_GROUPS, 32),
            nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(GROUP_NORM_GROUPS, 64),
            nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(GROUP_NORM_GROUPS, 128),
            nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(GROUP_NORM_GROUPS, 128),
            nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE),
        )
        self.encoder_hidden = nn.Sequential(
            nn.Flatten(), nn.Linear(ENCODED_FEATURES, 128),
            nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE),
        )
        self.mu_head = nn.Linear(128, LATENT_DIM)
        self.log_var_head = nn.Linear(128, LATENT_DIM)

        self.decoder_projection = nn.Linear(LATENT_DIM, ENCODED_FEATURES)
        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(GROUP_NORM_GROUPS, 128),
            nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(GROUP_NORM_GROUPS, 64),
            nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(GROUP_NORM_GROUPS, 32),
            nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE),
            nn.ConvTranspose2d(32, 22, kernel_size=4, stride=2, padding=1),
        )

        self.fc_tail = nn.Sequential(
            nn.Linear(2 * LATENT_DIM, 64),
            nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE),
            nn.Dropout(p=FC_TAIL_DROPOUT),
            nn.Linear(64, 32),
            nn.LeakyReLU(LEAKY_RELU_NEGATIVE_SLOPE),
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
        projected = self.decoder_projection(z).reshape(-1, *ENCODED_SHAPE)
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
        "architecture_version": ARCHITECTURE_VERSION,
        "input_shape": list(INPUT_SHAPE),
        "source_width": SOURCE_WIDTH,
        "dropped_leading_columns": DROPPED_LEADING_COLUMNS,
        "latent_dim": LATENT_DIM,
        "normalization": "GroupNorm",
        "group_norm_groups": GROUP_NORM_GROUPS,
        "group_norm_eps": 1e-5,
        "group_norm_affine": True,
        "encoder_group_norm_channels": [32, 64, 128, 128],
        "decoder_group_norm_channels": [128, 64, 32],
        "kernel_size": [4, 4],
        "stride": [2, 2],
        "padding": [1, 1],
        "encoder_channels": [22, 32, 64, 128, 128],
        "encoder_output_shapes": [[32, 8, 88], [64, 4, 44], [128, 2, 22], list(ENCODED_SHAPE)],
        "encoded_shape": list(ENCODED_SHAPE),
        "flattened_features": ENCODED_FEATURES,
        "encoder_hidden": [ENCODED_FEATURES, 128],
        "encoder_convolution_activation": "LeakyReLU",
        "encoder_hidden_activation": "LeakyReLU",
        "leaky_relu_negative_slope": LEAKY_RELU_NEGATIVE_SLOPE,
        "gaussian_parameter_head_activation": "linear",
        "decoder_projection": [LATENT_DIM, ENCODED_FEATURES],
        "decoder_projection_activation": "linear",
        "decoder_channels": [128, 128, 64, 32, 22],
        "decoder_output_shapes": [[128, 2, 22], [64, 4, 44], [32, 8, 88], list(INPUT_SHAPE)],
        "decoder_convolution_activation": "LeakyReLU after first three transposed convolutions",
        "group_norm_placement": "after all four encoder convolutions and first three decoder convolutions",
        "fc_tail_sizes": [64, 64, 32, 1],
        "fc_tail_hidden_activation": "LeakyReLU",
        "dropout": FC_TAIL_DROPOUT,
        "dropout_after_fc_layers": [1],
        "outputs": "reconstruction and preference logits",
    }
