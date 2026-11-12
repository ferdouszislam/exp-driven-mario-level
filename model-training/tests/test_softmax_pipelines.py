"""Check categorical tile reconstruction alongside an independent binary path."""

from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
BASE_ARCHITECTURES = ("initial_model_architecture", "model_arch_1", "model_arch_2")
sys.path.insert(0, str(TRAINING_ROOT))

from categorical_reconstruction import (
    reconstruction_loss_components,
    reconstruction_probabilities,
)
from reconstruction_previews import save_checkpoint_reconstructions


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def pipeline(base: str):
    """Resolve each folder's flat model import without polluting other tests."""
    folder = TRAINING_ROOT / (base + "_softmax")
    model_name, train_name = f"test_{base}_softmax_model", f"test_{base}_softmax_train"
    model = load_module(folder / "model.py", model_name)
    saved_path = sys.path[:]
    try:
        sys.path.insert(0, str(folder))
        with patch.dict(sys.modules, {"model": model}):
            training = load_module(folder / "train.py", train_name)
        yield model, training
    finally:
        sys.path[:] = saved_path
        sys.modules.pop(model_name, None)
        sys.modules.pop(train_name, None)


def binary_input(batch: int = 2, height: int = 3, width: int = 4):
    target = torch.zeros(batch, 22, height, width)
    target[:, 2] = 1
    target[:, 21, :, ::2] = 1
    return target


class CategoricalReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_tile_probabilities_normalize_separately_from_path(self):
        logits = torch.randn(2, 22, 3, 4)
        probabilities = reconstruction_probabilities(logits)
        self.assertEqual(probabilities.shape, logits.shape)
        torch.testing.assert_close(probabilities[:, :21].sum(1), torch.ones(2, 3, 4))
        torch.testing.assert_close(probabilities[:, 21], torch.sigmoid(logits[:, 21]))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

        # The path may coexist with any tile and never competes in its softmax.
        changed = logits.clone()
        changed[:, 21] = 10000
        changed_probabilities = reconstruction_probabilities(changed)
        torch.testing.assert_close(probabilities[:, :21], changed_probabilities[:, :21])
        torch.testing.assert_close(changed_probabilities[:, 21], torch.ones(2, 3, 4))
        torch.testing.assert_close(changed_probabilities.sum(1), torch.full((2, 3, 4), 2.0))

    def test_uniform_loss_uses_one_categorical_tile_and_one_binary_path_per_cell(self):
        for shape in ((1, 22, 1, 1), (3, 22, 4, 7)):
            with self.subTest(shape=shape):
                target = binary_input(shape[0], shape[2], shape[3])
                parts = reconstruction_loss_components(torch.zeros(shape), target)
                self.assertAlmostEqual(parts["tile_cross_entropy"].item(), math.log(21), places=6)
                self.assertAlmostEqual(parts["path_bce"].item(), math.log(2), places=6)

    def test_extreme_incorrect_logits_have_finite_losses_and_gradients(self):
        logits = torch.full((1, 22, 1, 2), -10000.0)
        logits[:, 0] = 10000  # Wrong tile, including at a visited cell.
        logits[:, 21, :, 0] = -10000
        logits[:, 21, :, 1] = 10000
        logits.requires_grad_()
        target = binary_input(1, 1, 2)
        parts = reconstruction_loss_components(logits, target)
        loss = parts["tile_cross_entropy"] + parts["path_bce"]
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 20000)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[:, :21].abs().sum().item(), 0)
        self.assertGreater(logits.grad[:, 21].abs().sum().item(), 0)

    def test_tile_and_path_losses_have_separate_output_gradients(self):
        logits = torch.zeros(2, 22, 3, 4, requires_grad=True)
        parts = reconstruction_loss_components(logits, binary_input())
        tile_gradient = torch.autograd.grad(parts["tile_cross_entropy"], logits, retain_graph=True)[0]
        path_gradient = torch.autograd.grad(parts["path_bce"], logits)[0]
        self.assertEqual(tile_gradient[:, 21].abs().sum().item(), 0)
        self.assertEqual(path_gradient[:, :21].abs().sum().item(), 0)
        self.assertGreater(tile_gradient[:, :21].abs().sum().item(), 0)
        self.assertGreater(path_gradient[:, 21].abs().sum().item(), 0)

    def test_categorical_preview_activates_probabilities_and_keeps_path_independent(self):
        logits = torch.full((1, 22, 16, 176), -10.0)
        logits[:, 6] = 2  # The coin wins among tiles even with a larger path logit.
        logits[:, 21] = 1000
        logits[:, 21, 0, 0] = -1000
        logits[:, 21, 0, 1] = 0

        class ControlledVAE(torch.nn.Module):
            def encode(self, inputs):
                return torch.zeros(1, 32), torch.zeros(1, 32)

            def reparameterize(self, mu, log_var):
                return mu

            def decode(self, latent):
                return logits

        model = ControlledVAE()
        data = SimpleNamespace(tensors={"validation/input.npy": binary_input(1, 16, 176)[0]}, channels={})
        samples = [{"encoding": "validation/input.npy", "level_id": "test-level"}]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preview"
            with patch("reconstruction_previews._renderer", return_value=(
                    lambda array, **kwargs: Image.new("RGB", (176, 16)))), patch(
                    "reconstruction_previews.reconstruction_probabilities",
                    wraps=reconstruction_probabilities) as activation:
                manifest = save_checkpoint_reconstructions(
                    model, data, samples, output, torch.device("cpu"), 42,
                    "epoch_001.pt", 1, categorical_tiles=True)
            activation.assert_called_once()
            generated = np.load(output / manifest["samples"][0]["generated_encoding"], allow_pickle=False)
            self.assertTrue(generated[6].all())
            np.testing.assert_array_equal(generated[:21].sum(0), np.ones((16, 176), dtype=np.uint8))
            self.assertEqual(generated[21, 0, 0], 0)
            self.assertEqual(generated[21, 0, 1], 1)
            self.assertEqual(generated[21, 1, 1], 1)
            self.assertIn("softmax", manifest["decoding"].lower())
            self.assertIn("sigmoid", manifest["decoding"].lower())
            self.assertTrue(model.training)


class SoftmaxArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_variants_preserve_base_networks_and_identify_their_output_convention(self):
        for base in BASE_ARCHITECTURES:
            with self.subTest(architecture=base), pipeline(base) as (model, training):
                original_name = f"test_{base}_original"
                original = load_module(TRAINING_ROOT / base / "model.py", original_name)
                try:
                    torch.manual_seed(42)
                    base_network = original.PairwiseCNNVAE()
                    torch.manual_seed(42)
                    network = model.PairwiseCNNVAE()
                    self.assertEqual(model.architecture_metadata()["architecture_version"], base + "_crop16_softmax")
                    self.assertEqual(original.architecture_metadata()["architecture_version"], base + "_crop16")
                    self.assertEqual(list(network.state_dict()), list(base_network.state_dict()))
                    for key, parameter in network.state_dict().items():
                        torch.testing.assert_close(parameter, base_network.state_dict()[key])
                    self.assertEqual(training.DEFAULT_HYPERPARAMETERS.parent.name, base + "_softmax")
                    self.assertTrue(training.DEFAULT_HYPERPARAMETERS.is_file())
                    self.assertEqual(training.DEFAULT_DATASET,
                                     TRAINING_ROOT.parent / "datasets/fun-study-dataset-analysis/preprocessed_dataset")
                finally:
                    sys.modules.pop(original_name, None)

    def test_each_decoder_exposes_softmax_tiles_and_sigmoid_path(self):
        known_logits = torch.zeros(1, 22, 2, 3)
        known_logits[:, 6] = math.log(20)
        known_logits[:, 21] = math.log(3)
        for base in BASE_ARCHITECTURES:
            with self.subTest(architecture=base), pipeline(base) as (model, _):
                network = model.PairwiseCNNVAE().eval()
                latent = torch.zeros(1, 32)
                with patch.object(network, "decode", return_value=known_logits) as decode:
                    probabilities = network.decode_probabilities(latent)
                decode.assert_called_once_with(latent)
                # Twenty competing tile channels each have exp(logit)=1.
                torch.testing.assert_close(probabilities[:, 6], torch.full((1, 2, 3), 0.5))
                torch.testing.assert_close(probabilities[:, 21], torch.full((1, 2, 3), 0.75))
                torch.testing.assert_close(probabilities[:, :21].sum(1), torch.ones(1, 2, 3))

    def test_each_training_loss_records_components_and_weights_reconstruction_once(self):
        target = binary_input()
        for base in BASE_ARCHITECTURES:
            with self.subTest(architecture=base), pipeline(base) as (_, training):
                output = {
                    "reconstruction_logits_a": torch.zeros_like(target),
                    "reconstruction_logits_b": torch.zeros_like(target),
                    "mu_a": torch.ones(2, 32), "mu_b": torch.zeros(2, 32),
                    "log_var_a": torch.zeros(2, 32), "log_var_b": torch.zeros(2, 32),
                    "rank_logits": torch.zeros(2),
                }
                parts = training.loss_components(output, target, target, torch.tensor([0., 1.]))
                self.assertAlmostEqual(parts["tile_cross_entropy"].item(), math.log(21), places=6)
                self.assertAlmostEqual(parts["path_bce"].item(), math.log(2), places=6)
                self.assertAlmostEqual(parts["reconstruction_loss"].item(), math.log(21) + math.log(2), places=6)
                self.assertAlmostEqual(parts["kl_divergence"].item(), 8.0, places=6)
                self.assertAlmostEqual(parts["ranking_bce"].item(), math.log(2), places=6)
                hyperparameters = training.Hyperparameters(1e-4, 2, 3, 5)
                expected = 2 * (math.log(21) + math.log(2)) + 3 * 8 + 5 * math.log(2)
                self.assertAlmostEqual(training.weighted_loss(parts, hyperparameters).item(), expected, places=5)
                self.assertNotIn("reconstruction_bce", parts)

    def test_each_decoder_learns_both_tile_and_path_outputs(self):
        for base in BASE_ARCHITECTURES:
            with self.subTest(architecture=base), pipeline(base) as (model, _):
                torch.manual_seed(19)
                network = model.PairwiseCNNVAE().eval()
                logits = network.decode(torch.randn(2, 32))
                self.assertEqual(tuple(logits.shape), (2, 22, 16, 176))
                target = binary_input(2, 16, 176)
                parts = reconstruction_loss_components(logits, target)
                (parts["tile_cross_entropy"] + parts["path_bce"]).backward()
                # Transposed-convolution weights have output channels on axis 1.
                final_gradient = network.decoder_convs[-1].weight.grad
                self.assertTrue(torch.isfinite(final_gradient).all())
                self.assertGreater(final_gradient[:, :21].abs().sum().item(), 0)
                self.assertGreater(final_gradient[:, 21:].abs().sum().item(), 0)
                self.assertGreater(network.decoder_projection[0].weight.grad.abs().sum().item()
                                   if isinstance(network.decoder_projection, torch.nn.Sequential)
                                   else network.decoder_projection.weight.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
