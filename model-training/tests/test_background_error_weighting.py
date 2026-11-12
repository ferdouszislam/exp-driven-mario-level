"""Check the asymmetric background-error penalty and its reconstruction gradients."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from background_error_weighting import (
    background_error_weights,
    weighted_bce_reconstruction,
    weighted_categorical_reconstruction_loss_components,
)
from categorical_reconstruction import reconstruction_loss_components


def binary_target(tile_ids, path=None):
    """Build valid one-hot tiles plus an independent path channel."""
    tiles = torch.tensor(tile_ids, dtype=torch.long)
    tile_bits = F.one_hot(tiles, num_classes=21).permute(0, 3, 1, 2).double()
    path_bits = torch.zeros_like(tiles, dtype=torch.float64)
    if path is not None:
        path_bits = torch.tensor(path, dtype=torch.float64)
    return torch.cat((tile_bits, path_bits.unsqueeze(1)), dim=1)


class BackgroundErrorWeightingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gate_is_asymmetric_and_includes_both_solid_symbols(self):
        # Background mistakes and mistakes between other tile types stay at 1.
        # | is a mushroom stalk, so it remains a non-background tile.
        target = binary_target([[[0, 19, 2, 6, 6]], [[6, 6, 6, 20, 6]]])
        predicted = torch.tensor([[[2, 0, 6, 0, 2]], [[19, 7, 6, 2, 20]]])
        logits = torch.full_like(target, -1)
        logits[:, :21].scatter_(1, predicted.unsqueeze(1), 2)
        logits.requires_grad_()
        weights = background_error_weights(logits, target, 3.5)
        torch.testing.assert_close(weights, torch.tensor(
            [[[1, 1, 1, 3.5, 3.5]], [[3.5, 1, 1, 3.5, 1]]],
            dtype=torch.float64,
        ))
        self.assertFalse(weights.requires_grad)

    def test_path_logits_and_path_targets_do_not_change_gate(self):
        target = binary_target([[[6, 2]]])
        logits = torch.zeros_like(target)
        original = background_error_weights(logits, target, 4)
        logits[:, 21] = torch.tensor([[[-10000, 10000]]])
        target[:, 21] = 1
        torch.testing.assert_close(background_error_weights(logits, target, 4), original)
        torch.testing.assert_close(original, torch.tensor([[[4., 1.]]], dtype=torch.float64))

    def test_bce_exact_loss_and_gradients_keep_fixed_22_channel_denominator(self):
        target = binary_target([[[0, 6, 20]]], [[[0, 1, 0]]])
        logits = torch.zeros_like(target, requires_grad=True)
        multiplier = 4
        loss = weighted_bce_reconstruction(logits, target, multiplier)
        # Every raw BCE term is ln(2); only the last two cells' 21 tile terms grow.
        expected = math.log(2) * (21 * (1 + 2 * multiplier) + 3) / 66
        self.assertAlmostEqual(loss.item(), expected, places=12)
        loss.backward()
        expected_gradient = (0.5 - target) / 66
        expected_gradient[:, :21, :, 1:] *= multiplier
        torch.testing.assert_close(logits.grad, expected_gradient)

    def test_categorical_exact_loss_and_gradients_keep_path_unweighted(self):
        target = binary_target([[[0, 6, 20]]], [[[0, 1, 0]]])
        logits = torch.zeros_like(target, requires_grad=True)
        multiplier = 4
        parts = weighted_categorical_reconstruction_loss_components(logits, target, multiplier)
        self.assertAlmostEqual(parts["tile_cross_entropy"].item(),
                               math.log(21) * (1 + 2 * multiplier) / 3, places=12)
        self.assertAlmostEqual(parts["path_bce"].item(), math.log(2), places=12)
        (parts["tile_cross_entropy"] + parts["path_bce"]).backward()
        expected_gradient = torch.empty_like(target)
        expected_gradient[:, :21] = (1 / 21 - target[:, :21]) / 3
        expected_gradient[:, :21, :, 1:] *= multiplier
        expected_gradient[:, 21] = (0.5 - target[:, 21]) / 3
        torch.testing.assert_close(logits.grad, expected_gradient)

    def test_weight_one_matches_existing_bce_and_categorical_losses_and_gradients(self):
        torch.manual_seed(23)
        target = binary_target([[[6, 2, 19], [0, 20, 1]], [[3, 6, 0], [2, 19, 20]]])
        initial = torch.randn_like(target)
        for categorical in (False, True):
            with self.subTest(categorical=categorical):
                logits = initial.clone().requires_grad_()
                reference = initial.clone().requires_grad_()
                if categorical:
                    actual_parts = weighted_categorical_reconstruction_loss_components(logits, target, 1)
                    expected_parts = reconstruction_loss_components(reference, target)
                    for name in expected_parts:
                        torch.testing.assert_close(actual_parts[name], expected_parts[name])
                    actual, expected = sum(actual_parts.values()), sum(expected_parts.values())
                else:
                    actual = weighted_bce_reconstruction(logits, target, 1)
                    expected = F.binary_cross_entropy_with_logits(reference, target)
                torch.testing.assert_close(actual, expected)
                actual.backward()
                expected.backward()
                torch.testing.assert_close(logits.grad, reference.grad)

    def test_extreme_logits_keep_losses_and_gradients_finite(self):
        target = binary_target([[[6, 20]]], [[[1, 0]]])
        initial = torch.full_like(target, -10000)
        initial[:, 0] = 10000
        initial[:, 21, :, 1] = 10000
        for categorical in (False, True):
            with self.subTest(categorical=categorical):
                logits = initial.clone().requires_grad_()
                if categorical:
                    parts = weighted_categorical_reconstruction_loss_components(logits, target, 5)
                    loss = sum(parts.values())
                else:
                    loss = weighted_bce_reconstruction(logits, target, 5)
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(loss.item(), 0)
                loss.backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertGreater(logits.grad[:, :21].abs().sum().item(), 0)
                self.assertGreater(logits.grad[:, 21].abs().sum().item(), 0)

    def test_invalid_weights_are_rejected_by_every_entrypoint(self):
        target = binary_target([[[6]]])
        logits = torch.zeros_like(target)
        for weight in (0, -1, 0.99, math.nan, math.inf, -math.inf):
            for function in (background_error_weights, weighted_bce_reconstruction,
                             weighted_categorical_reconstruction_loss_components):
                with self.subTest(weight=weight, function=function.__name__):
                    with self.assertRaisesRegex(ValueError, "finite and at least 1"):
                        function(logits, target, weight)


if __name__ == "__main__":
    unittest.main()
