"""Check validation sampling and checkpoint reconstruction images."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn


TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

from data import INPUT_SHAPE, TILE_SYMBOLS
from reconstruction_previews import select_validation_levels, save_checkpoint_reconstructions


class ControlledVAE(nn.Module):
    """Return known tile/path logits after an observable posterior sample."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.decoded_latent = None

    def _record(self, name):
        self.calls.append((name, self.training, torch.is_grad_enabled()))

    def encode(self, inputs):
        self._record("encode")
        mu = torch.full((len(inputs), 32), 2.0, device=inputs.device)
        return mu, torch.zeros_like(mu)

    def reparameterize(self, mu, log_var):
        self._record("reparameterize")
        return mu + 1

    def decode(self, latent):
        self._record("decode")
        self.decoded_latent = latent.detach().clone()
        logits = torch.full((len(latent), *INPUT_SHAPE), -10.0, device=latent.device)
        logits[:, 6] = 2.0  # Coin must win among geometry channels.
        logits[:, 2] = 1.0
        logits[:, 21] = 1000.0  # A larger path logit must not replace geometry.
        logits[:, 21, 0, 0] = -1000.0
        logits[:, 21, 0, 1] = 0.0  # sigmoid(0) == 0.5 is included.
        return logits


class ReconstructionPreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.entries, tensors = [], {}
        for comparison in range(4):
            plays = []
            for side in range(2):
                level_id = f"level-{comparison}-{side}"
                encoding = f"validation_set/comparison-{comparison}/{level_id}.npy"
                plays.append({"level_id": level_id, "playthrough_key": f"participant_{level_id}",
                              "files": {"encoding": encoding}})
                inputs = torch.zeros(INPUT_SHAPE)
                inputs[2] = 1
                inputs[21, 2, 3] = 1
                tensors[encoding] = inputs
            self.entries.append({"instance_id": f"comparison-{comparison}",
                                 "split": "validation_set", "playthroughs": plays})
        channels = {
            "shape": list(INPUT_SHAPE), "dtype": "uint8",
            "axis_order": ["channel", "row", "column"], "path_channel_index": 21,
            "channels": [{"index": index, "symbol": symbol,
                          "name": "PATH" if index == 21 else symbol}
                         for index, symbol in enumerate(TILE_SYMBOLS + ["P"])],
        }
        self.data = SimpleNamespace(tensors=tensors, channels=channels)

    def test_seeded_sampling_chooses_five_distinct_validation_levels(self):
        duplicate = {"instance_id": "zz-repeated-comparison", "split": "validation_set",
                     "playthroughs": [self.entries[0]["playthroughs"][0]]}
        entries = self.entries + [duplicate]
        random.seed(321)
        random_state = random.getstate()
        samples = select_validation_levels(entries, seed=42)
        self.assertEqual(random.getstate(), random_state)
        self.assertEqual(len(samples), 5)
        self.assertEqual(len({sample["level_id"] for sample in samples}), 5)
        self.assertTrue(all(sample["encoding"].startswith("validation_set/") for sample in samples))
        self.assertTrue(all(sample["instance_id"] != duplicate["instance_id"] for sample in samples))
        self.assertEqual(samples, select_validation_levels(list(reversed(entries)), seed=42))
        self.assertNotEqual(samples, select_validation_levels(entries, seed=43))

    def test_small_and_empty_validation_sets_do_not_repeat_inputs(self):
        samples = select_validation_levels(self.entries[:1], seed=42)
        self.assertEqual(len(samples), 2)
        self.assertEqual(len({sample["level_id"] for sample in samples}), 2)
        self.assertEqual(select_validation_levels([], seed=42), [])
        model = ControlledVAE()
        output = self.root / "empty"
        with patch("reconstruction_previews._renderer", side_effect=AssertionError("Unexpected renderer import")):
            manifest = save_checkpoint_reconstructions(
                model, self.data, [], output, torch.device("cpu"), 42, "epoch_001.pt", 1,
            )
        self.assertEqual(manifest["sample_count"], 0)
        self.assertIsNone(manifest["comparison_image"])
        self.assertIn("skipped_reason", manifest)
        self.assertEqual(json.loads((output / "samples.json").read_text()), manifest)
        self.assertEqual([path.name for path in output.iterdir()], ["samples.json"])
        self.assertEqual(model.calls, [])

    def test_saved_images_and_encodings_use_sampled_vae_output(self):
        model = ControlledVAE()
        samples = select_validation_levels(self.entries, seed=42)
        output = self.root / "epoch_003"
        manifest = save_checkpoint_reconstructions(
            model, self.data, samples, output, torch.device("cpu"), 43, "checkpoints/epoch_003.pt", 3,
        )
        self.assertTrue(model.training)
        self.assertEqual([call[0] for call in model.calls], ["encode", "reparameterize", "decode"])
        self.assertTrue(all(not training and not gradients for _, training, gradients in model.calls))
        torch.testing.assert_close(model.decoded_latent, torch.full((5, 32), 3.0))
        self.assertEqual(manifest["sample_count"], 5)
        self.assertEqual(manifest["checkpoint"], "checkpoints/epoch_003.pt")
        self.assertEqual(manifest["epoch"], 3)
        self.assertEqual(manifest["seed"], 43)
        self.assertEqual(json.loads((output / "samples.json").read_text()), manifest)
        self.assertEqual(json.loads((output / "channels.json").read_text()), self.data.channels)
        for source, saved in zip(samples, manifest["samples"], strict=True):
            self.assertEqual(saved["level_id"], source["level_id"])
            self.assertEqual(saved["encoding"], source["encoding"])
            generated = np.load(output / saved["generated_encoding"], allow_pickle=False)
            self.assertEqual(generated.shape, INPUT_SHAPE)
            self.assertEqual(generated.dtype, np.uint8)
            np.testing.assert_array_equal(generated[:21].sum(0), np.ones(INPUT_SHAPE[1:], dtype=np.uint8))
            self.assertTrue(generated[6].all())
            self.assertEqual(generated[21, 0, 0], 0)
            self.assertEqual(generated[21, 0, 1], 1)
            self.assertEqual(generated[21, 2, 3], 1)
            images = []
            for field in ("input_image", "generated_image"):
                with Image.open(output / saved[field]) as image:
                    self.assertEqual(image.size, (2816, 256))
                    self.assertEqual(image.format, "PNG")
                    images.append(np.asarray(image).copy())
            self.assertFalse(np.array_equal(images[0], images[1]))
        with Image.open(output / manifest["comparison_image"]) as comparison:
            self.assertGreaterEqual(comparison.width, 2816)
            self.assertGreaterEqual(comparison.height, 5 * 2 * 256)

    def test_rendering_failure_restores_training_mode(self):
        model = ControlledVAE()
        samples = select_validation_levels(self.entries, seed=42, count=1)
        for previous_training in (True, False):
            with self.subTest(previous_training=previous_training):
                model.train(previous_training)
                with patch.object(model, "decode", side_effect=RuntimeError("decoder failure")):
                    with self.assertRaisesRegex(RuntimeError, "decoder failure"):
                        save_checkpoint_reconstructions(
                            model, self.data, samples, self.root / "failed", torch.device("cpu"), 42,
                            "epoch_001.pt", 1,
                        )
                self.assertEqual(model.training, previous_training)


if __name__ == "__main__":
    unittest.main()
