"""Exercise level generation as an independent pass over saved checkpoints."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_architecture_pipelines import INPUT_SHAPE, write_dataset
from test_weighted_pipelines import TRAINING_ROOT, pipeline

sys.path.insert(0, str(TRAINING_ROOT))

from data import load_training_data
from generate_levels import publish_outputs
from reconstruction_previews import save_generated_levels


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / "dataset"
        write_dataset(self.dataset)
        manifest_path = self.dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for source in list(manifest["instances"]):
            if source["split"] == "test_set":
                continue
            for index in (1, 2):
                entry = copy.deepcopy(source)
                entry["instance_id"] += f"-{index}"
                entry["split_group_id"] = entry["instance_id"]
                relative = f"{entry['split']}/{entry['instance_id']}"
                entry["relative_directory"] = relative
                entry["ranking_file"] = f"{relative}/ranking.txt"
                directory = self.dataset / relative
                directory.mkdir()
                levels = []
                for source_play, play in zip(source["playthroughs"], entry["playthroughs"], strict=True):
                    play["level_id"] += f"-{index}"
                    play["playthrough_key"] += f"-{index}"
                    play["files"]["encoding"] = f"{relative}/{play['level_id']}.npy"
                    levels.append(play["level_id"])
                    np.save(self.dataset / play["files"]["encoding"],
                            np.load(self.dataset / source_play["files"]["encoding"], allow_pickle=False),
                            allow_pickle=False)
                entry["preferred_level_id"], entry["less_preferred_level_id"] = levels
                (directory / "ranking.txt").write_text("\n".join(levels) + "\n")
                manifest["instances"].append(entry)
        manifest_path.write_text(json.dumps(manifest))
        self.data = load_training_data(self.dataset, None, 42)

    def write_checkpoint(self, folder, name="model_arch_1", epoch=1, *, fitting=None, validation=None):
        folder.mkdir(parents=True, exist_ok=True)
        with pipeline(name) as (module, _):
            torch.manual_seed(123)
            model = module.PairwiseCNNVAE()
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": {"dataset": str(self.dataset), "seed": 42,
                           "architecture": module.architecture_metadata(),
                           "dataset_fingerprint": self.data.dataset_fingerprint},
                "channels": self.data.channels,
                "selected_instance_ids": [entry["instance_id"] for entry in self.data.entries],
                "fitting_instance_ids": ([entry["instance_id"] for entry in self.data.entries]
                                         if fitting is None else fitting),
                "validation_instance_ids": ([entry["instance_id"] for entry in self.data.validation_entries]
                                            if validation is None else validation),
            }
        path = folder / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        return path

    def run_generation(self, *arguments, expected_success=True):
        result = subprocess.run(
            [sys.executable, str(TRAINING_ROOT / "generate_levels.py"),
             "--device", "cpu", "--cpu-threads", "2", *map(str, arguments)],
            cwd=self.root, capture_output=True, text=True, timeout=120,
        )
        if expected_success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_recursive_pass_handles_every_checkpoint_and_reruns_deterministically(self):
        runs = self.root / "runs"
        first = runs / "model-a/run/parameters/checkpoints"
        second = runs / "model-b/run/parameters/checkpoints"
        checkpoints = [self.write_checkpoint(first, epoch=1), self.write_checkpoint(first, epoch=3),
                       self.write_checkpoint(second, "model_arch_2_softmax_weighted", epoch=2)]
        self.run_generation("--runs-dir", runs)
        snapshots = {}
        for checkpoint in checkpoints:
            outputs = checkpoint.parent.parent / "reconstructions"
            for portion in ("train", "validation", "generated"):
                destination = outputs / portion / checkpoint.stem
                record = json.loads((destination / "samples.json").read_text())
                self.assertEqual(record["sample_count"], 5)
                self.assertEqual(record["split"], portion)
                self.assertEqual(len(list(destination.glob("*_generated.png"))), 5)
                self.assertEqual(len(list(destination.glob("*_generated.npy"))), 5)
                self.assertGreater((destination / "comparison.png").stat().st_size, 0)
                for encoding in destination.glob("*_generated.npy"):
                    array = np.load(encoding, allow_pickle=False)
                    self.assertEqual(array.shape, INPUT_SHAPE)
                    self.assertEqual(array.dtype, np.uint8)
                    np.testing.assert_array_equal(array[:21].sum(0), np.ones(INPUT_SHAPE[1:]))
                    snapshots[encoding] = encoding.read_bytes()
                if portion != "generated":
                    expected = "train_set/" if portion == "train" else "validation_set/"
                    self.assertTrue(all(sample["encoding"].startswith(expected)
                                        for sample in record["samples"]))
                    self.assertEqual(len({sample["level_id"] for sample in record["samples"]}), 5)
            self.assertFalse((outputs / "epoch_001").exists())
        self.run_generation("--runs-dir", runs)
        for path, contents in snapshots.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_saved_membership_controls_inputs_and_missing_validation_is_skipped(self):
        folder = self.root / "run/parameters/checkpoints"
        fitting = [self.data.entries[0]["instance_id"]]
        checkpoint = self.write_checkpoint(folder, fitting=fitting, validation=[])
        # A saved cluster path can be overridden without rewriting the checkpoint.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        saved["config"]["dataset"] = "/unavailable/cluster/export"
        torch.save(saved, checkpoint)
        self.run_generation("--checkpoints-dir", folder, "--dataset", self.dataset)
        outputs = folder.parent / "reconstructions"
        training = json.loads((outputs / "train/epoch_001/samples.json").read_text())
        self.assertEqual(training["sample_count"], 2)
        self.assertEqual({sample["instance_id"] for sample in training["samples"]}, set(fitting))
        validation = json.loads((outputs / "validation/epoch_001/samples.json").read_text())
        self.assertEqual(validation["sample_count"], 0)
        self.assertIn("skipped_reason", validation)
        self.assertFalse((outputs / "validation/epoch_001/comparison.png").exists())
        generated = json.loads((outputs / "generated/epoch_001/samples.json").read_text())
        self.assertEqual(generated["sample_count"], 5)

    def test_invalid_model_state_does_not_replace_previous_outputs(self):
        folder = self.root / "run/parameters/checkpoints"
        checkpoint = self.write_checkpoint(folder)
        self.run_generation("--checkpoints-dir", folder)
        generated = folder.parent / "reconstructions/generated/epoch_001/01_generated.npy"
        before = generated.read_bytes()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        saved["model_state_dict"].pop(next(iter(saved["model_state_dict"])))
        torch.save(saved, checkpoint)
        self.run_generation("--checkpoints-dir", folder, expected_success=False)
        self.assertEqual(generated.read_bytes(), before)

    def test_failed_index_publication_restores_every_previous_output(self):
        output = self.root / "reconstructions"
        staging = self.root / "staging"
        staging.mkdir()
        for split in ("train", "validation", "generated"):
            destination = output / split / "epoch_001"
            destination.mkdir(parents=True)
            (destination / "samples.json").write_text(json.dumps({"split": split}))
            (destination / "old.txt").write_text(f"previous {split}")
            (staging / split).mkdir()
            (staging / split / "samples.json").write_text(json.dumps({"split": split}))
            (staging / split / "new.txt").write_text(f"replacement {split}")
        previous_index = '[{"checkpoint": "checkpoints/epoch_001.pt", "old": true}]\n'
        (output / "index.json").write_text(previous_index)
        original_replace = Path.replace

        def fail_index_publication(source, destination):
            if source == staging / "index.json" and Path(destination) == output / "index.json":
                raise OSError("simulated index publication failure")
            return original_replace(source, destination)

        with patch.object(Path, "replace", new=fail_index_publication):
            with self.assertRaisesRegex(OSError, "simulated index publication failure"):
                publish_outputs(staging, output, "epoch_001", [{"checkpoint": "replacement"}])
        self.assertEqual((output / "index.json").read_text(), previous_index)
        for split in ("train", "validation", "generated"):
            destination = output / split / "epoch_001"
            self.assertEqual((destination / "old.txt").read_text(), f"previous {split}")
            self.assertEqual(json.loads((destination / "samples.json").read_text()), {"split": split})
            self.assertFalse((destination / "new.txt").exists())

    def test_prior_generation_uses_decoder_only_and_records_latents(self):
        class DecoderOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.latents = None

            def encode(self, inputs):
                raise AssertionError("Prior sampling must not encode any input")

            def decode(self, latent):
                self.assertion = (self.training, torch.is_grad_enabled())
                self.latents = latent.detach().clone()
                logits = torch.zeros(len(latent), *INPUT_SHAPE)
                logits[:, 6] = 10
                return logits

        model = DecoderOnly()
        output = self.root / "generated"
        seed, latent_dim = 41, 32
        record = save_generated_levels(model, self.data.channels, output, torch.device("cpu"),
                                       seed, "checkpoints/epoch_001.pt", 1, latent_dim=latent_dim)
        self.assertEqual(model.assertion, (False, False))
        self.assertTrue(model.training)
        expected = torch.randn(5, latent_dim, generator=torch.Generator().manual_seed(seed))
        torch.testing.assert_close(model.latents, expected)
        self.assertEqual(record["split"], "generated")
        self.assertEqual(record["sample_count"], 5)
        for index in range(5):
            latent = np.load(output / f"{index + 1:02d}_latent.npy", allow_pickle=False)
            np.testing.assert_array_equal(latent, expected[index].numpy())


if __name__ == "__main__":
    unittest.main()
