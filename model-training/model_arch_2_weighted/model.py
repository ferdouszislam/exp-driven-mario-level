"""Network from ../model_arch_2.md with weighted reconstruction training metadata."""

from __future__ import annotations

import torch
from torch import Tensor, nn


SOURCE_WIDTH = 192
DROPPED_LEADING_COLUMNS = 16
INPUT_SHAPE = (22, 16, SOURCE_WIDTH - DROPPED_LEADING_COLUMNS)
LATENT_DIM = 32
ARCHITECTURE_VERSION = "model_arch_2_crop16_weighted"
ENCODED_SHAPE = (128, 1, 11)
ENCODED_FEATURES = 1408
LAYER_NORM_EPS = 1e-5
FC_TAIL_DROPOUT = 0.2
ENCODER_LAYER_NORM_SHAPES = ((32, 8, 88), (64, 4, 44), (128, 2, 22), ENCODED_SHAPE)
DECODER_LAYER_NORM_SHAPES = ((128, 2, 22), (64, 4, 44), (32, 8, 88))


class PairwiseCNNVAE(nn.Module):
    """Share the VAE between both inputs and return reconstruction/rank logits."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder_convs = nn.Sequential(
            nn.Conv2d(22, 32, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(ENCODER_LAYER_NORM_SHAPES[0], eps=LAYER_NORM_EPS),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(ENCODER_LAYER_NORM_SHAPES[1], eps=LAYER_NORM_EPS),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(ENCODER_LAYER_NORM_SHAPES[2], eps=LAYER_NORM_EPS),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(ENCODER_LAYER_NORM_SHAPES[3], eps=LAYER_NORM_EPS),
            nn.ReLU(),
        )
        self.encoder_hidden = nn.Sequential(
            nn.Flatten(), nn.Linear(ENCODED_FEATURES, 128), nn.ReLU(),
        )
        self.mu_head = nn.Linear(128, LATENT_DIM)
        self.log_var_head = nn.Linear(128, LATENT_DIM)

        self.decoder_projection = nn.Linear(LATENT_DIM, ENCODED_FEATURES)
        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(DECODER_LAYER_NORM_SHAPES[0], eps=LAYER_NORM_EPS),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(DECODER_LAYER_NORM_SHAPES[1], eps=LAYER_NORM_EPS),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm(DECODER_LAYER_NORM_SHAPES[2], eps=LAYER_NORM_EPS),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 22, kernel_size=4, stride=2, padding=1),
        )
        self.fc_tail = nn.Sequential(
            nn.Linear(2 * LATENT_DIM, 64), nn.ReLU(),
            nn.Dropout(p=FC_TAIL_DROPOUT),
            nn.Linear(64, 32), nn.ReLU(),
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
            "mu_a": mu_a, "mu_b": mu_b,
            "log_var_a": log_var_a, "log_var_b": log_var_b,
            "z_a": z_a, "z_b": z_b,
            "rank_logits": self.preference_logits(z_a, z_b),
        }


def architecture_metadata() -> dict[str, object]:
    """Return this variant's fixed choices to save beside each checkpoint."""
    return {
        "model_class": "PairwiseCNNVAE",
        "architecture_version": ARCHITECTURE_VERSION,
        "input_shape": list(INPUT_SHAPE),
        "source_width": SOURCE_WIDTH,
        "dropped_leading_columns": DROPPED_LEADING_COLUMNS,
        "latent_dim": LATENT_DIM,
        "normalization": "LayerNorm",
        "layer_norm_axes": ["channel", "height", "width"],
        "layer_norm_eps": LAYER_NORM_EPS,
        "layer_norm_elementwise_affine": True,
        "encoder_layer_norm_shapes": [list(shape) for shape in ENCODER_LAYER_NORM_SHAPES],
        "decoder_layer_norm_shapes": [list(shape) for shape in DECODER_LAYER_NORM_SHAPES],
        "kernel_size": [4, 4], "stride": [2, 2], "padding": [1, 1],
        "dilation": [1, 1], "output_padding": [0, 0],
        "encoder_channels": [22, 32, 64, 128, 128],
        "encoder_output_shapes": [list(shape) for shape in ENCODER_LAYER_NORM_SHAPES],
        "encoded_shape": list(ENCODED_SHAPE),
        "flattened_features": ENCODED_FEATURES,
        "encoder_hidden": [ENCODED_FEATURES, 128],
        "encoder_convolution_activation": "ReLU",
        "encoder_hidden_activation": "ReLU",
        "gaussian_parameter_head_activation": "linear",
        "decoder_projection": [LATENT_DIM, ENCODED_FEATURES],
        "decoder_projection_activation": "linear",
        "decoder_channels": [128, 128, 64, 32, 22],
        "decoder_output_shapes": [list(shape) for shape in DECODER_LAYER_NORM_SHAPES] + [list(INPUT_SHAPE)],
        "decoder_convolution_activation": "ReLU after first three transposed convolutions",
        "layer_norm_placement": "after all four encoder convolutions and first three decoder convolutions",
        "fc_tail_sizes": [64, 64, 32, 1],
        "fc_tail_hidden_activation": "ReLU",
        "dropout": FC_TAIL_DROPOUT,
        "dropout_after_fc_layers": [1],
        "background_error_weighting": {
            "hyperparameter": "background_error_weight",
            "minimum_weight": 1.0,
            "background_tile_symbols": ["X", "-", "#"],
            "background_tile_channel_indices": [0, 2, 19],
            "condition": "actual tile outside background AND tile argmax prediction inside background",
            "affected_loss": "all tile-channel errors at the selected cell; path excluded",
            "gate": "hard argmax, treated as constant during backpropagation",
            "normalization": "fixed cell/bit count, never the sum of weights",
        },
        "outputs": "reconstruction and preference logits",
    }
