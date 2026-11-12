"""Exercise all six conditional background-error training pipelines."""

from __future__ import annotations

from contextlib import contextmanager
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from test_architecture_pipelines import write_dataset


TRAINING_ROOT = Path(__file__).resolve().parents[1]
BASE_ARCHITECTURES = ("initial_model_architecture", "model_arch_1", "model_arch_2")
PIPELINES = tuple(base + suffix + "_weighted"
                  for suffix in ("", "_softmax") for base in BASE_ARCHITECTURES)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def pipeline(name: str):
    """Keep each folder's flat model import isolated during test discovery."""
    folder = TRAINING_ROOT / name
    model_name, train_name = f"test_{name}_model", f"test_{name}_train"
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


class ControlledVAE(torch.nn.Module):
    """Always decode a coin as ground so the conditional penalty must activate."""

    def __init__(self):
        super().__init__()
        logits = torch.zeros(1, 22, 1, 1)
        logits[:, 0] = 2
        self.logits = torch.nn.Parameter(logits)

    def forward(self, x_a, x_b):
        zero = self.logits.sum() * 0
        latent = torch.zeros(x_a.shape[0], 32) + zero
        return {
            "reconstruction_logits_a": self.logits.expand_as(x_a),
            "reconstruction_logits_b": self.logits.expand_as(x_b),
            "mu_a": latent, "mu_b": latent,
            "log_var_a": latent, "log_var_b": latent,
            "z_a": latent, "z_b": latent,
            "rank_logits": latent[:, 0],
        }

    def preference_logits(self, z_a, z_b):
        return z_a[:, 0] * 0


class WeightedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def run_command(self, arguments, cwd):
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=cwd,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_each_pipeline_preserves_its_base_network_and_has_a_distinct_version(self):
        for name in PIPELINES:
            base = name.removesuffix("_weighted")
            original_name = f"test_{base}_unweighted_model"
            with self.subTest(pipeline=name), pipeline(name) as (model, training):
                original = load_module(TRAINING_ROOT / base / "model.py", original_name)
                try:
                    self.assertEqual(model.architecture_metadata()["architecture_version"],
                                     original.architecture_metadata()["architecture_version"] + "_weighted")
                    torch.manual_seed(42)
                    before = original.PairwiseCNNVAE()
                    torch.manual_seed(42)
                    after = model.PairwiseCNNVAE()
                    self.assertEqual(list(before.state_dict()), list(after.state_dict()))
                    for key, value in before.state_dict().items():
                        torch.testing.assert_close(value, after.state_dict()[key])
                    self.assertEqual(training.HERE, TRAINING_ROOT / name)
                    self.assertEqual(training.DEFAULT_HYPERPARAMETERS.parent, TRAINING_ROOT / name)
                    self.assertTrue(training.DEFAULT_HYPERPARAMETERS.is_file())
                    self.assertTrue(training.load_hyperparameter_combinations(training.DEFAULT_HYPERPARAMETERS))
                finally:
                    sys.modules.pop(original_name, None)

    def test_fifth_hyperparameter_is_validated_and_distinguishes_output_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weights.txt"
            for name in PIPELINES:
                with self.subTest(pipeline=name), pipeline(name) as (_, training):
                    path.write_text("# five columns\n1e-4, 1, 0, 0, 1\n1e-4, 1, 0, 0, 3 # penalty\n")
                    combinations = training.load_hyperparameter_combinations(path)
                    self.assertEqual([hp.background_error_weight for hp in combinations], [1, 3])
                    self.assertNotEqual(*(training.hyperparameter_directory_name(hp) for hp in combinations))
                    for invalid in ("1e-4, 1, 1, 1", "1e-4, 1, 1, 1, 3, 4",
                                    "1e-4, 1, 1, 1, 0", "1e-4, 1, 1, 1, 0.5",
                                    "1e-4, 1, 1, 1, -3", "1e-4, 1, 1, 1, nan",
                                    "1e-4, 1, 1, 1, inf", "1e-4, 1, 1, 1, nope"):
                        with self.subTest(row=invalid):
                            path.write_text(invalid + "\n")
                            with self.assertRaises(ValueError):
                                training.load_hyperparameter_combinations(path)
                    path.write_text("1e-4, 1, 1, 1, 3\n1e-4, 1, 1, 1, 3\n")
                    with self.assertRaises(ValueError):
                        training.load_hyperparameter_combinations(path)

    def test_fitting_and_reporting_use_the_configured_penalty_weight(self):
        x = torch.zeros(1, 22, 1, 1)
        x[:, 6] = 1  # A coin, which ControlledVAE mistakes for ground.
        x[:, 21] = 1
        batch = (x, x, torch.ones(1))
        entry = {"instance_id": "pair", "preferred_level_id": "coin",
                 "less_preferred_level_id": "other"}
        for name in PIPELINES:
            with self.subTest(pipeline=name), pipeline(name) as (_, training):
                model = ControlledVAE()
                hp = training.Hyperparameters(1e-4, 1, 0, 0, 3)
                optimizer = torch.optim.SGD(model.parameters(), lr=0)
                logits = model.logits.detach()
                if "_softmax" in name:
                    metric = "reconstruction_loss"
                    expected = (3 * F.cross_entropy(logits[:, :21], x[:, :21].argmax(1))
                                + F.binary_cross_entropy_with_logits(logits[:, 21], x[:, 21]))
                else:
                    metric = "reconstruction_bce"
                    expected = (3 * F.binary_cross_entropy_with_logits(
                        logits[:, :21], x[:, :21], reduction="sum")
                        + F.binary_cross_entropy_with_logits(
                            logits[:, 21], x[:, 21], reduction="sum")) / x.numel()
                optimization = training.train_one_epoch(model, [batch], optimizer, hp, torch.device("cpu"))
                self.assertAlmostEqual(optimization[metric], expected.item(), places=6)
                self.assertAlmostEqual(optimization["total_loss"], expected.item(), places=6)
                with patch.object(training, "make_loader", return_value=[batch]):
                    reporting, _ = training.evaluate_training_pairs(
                        model, object(), [entry], hp, 1, torch.device("cpu"), 43)
                self.assertAlmostEqual(reporting[metric], expected.item(), places=6)
                self.assertAlmostEqual(reporting["total_loss"], expected.item(), places=6)
                self.assertEqual(reporting["normalized_hamming_distance"], 1)

    def test_all_six_cli_pipelines_save_restore_preview_and_analyze(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            write_dataset(dataset)
            # Include actual non-background tiles in both portions of the fixture.
            for encoding in dataset.glob("*/*/*.npy"):
                array = np.load(encoding, allow_pickle=False)
                array[:21, 4:6, 12:16] = 0
                array[6, 4:6, 12:16] = 1
                np.save(encoding, array, allow_pickle=False)
            parameters = root / "one-combination.txt"
            parameters.write_text("1e-4, 1, 1e-3, 1e-2, 3\n")
            for name in PIPELINES:
                with self.subTest(pipeline=name):
                    folder = TRAINING_ROOT / name
                    runs = root / name / "runs"
                    output = runs / "smoke"
                    self.run_command([
                        folder / "train.py", "--dataset", dataset,
                        "--hyperparameters-file", parameters,
                        "--epochs", "1", "--batch-size", "2", "--cpu-threads", "2",
                        "--device", "cpu", "--checkpoint-start-epoch", "1",
                        "--checkpoint-end-epoch", "1", "--output-dir", output,
                    ], root)
                    run_config = json.loads((output / "run_config.json").read_text())
                    self.assertEqual(run_config["hyperparameter_combinations"][0]["background_error_weight"], 3)
                    experiments = json.loads((output / "experiments.json").read_text())
                    self.assertEqual(experiments[0]["status"], "completed")
                    self.assertEqual(experiments[0]["hyperparameters"]["background_error_weight"], 3)
                    combination = output / experiments[0]["directory"]
                    config = json.loads((combination / "config.json").read_text())
                    self.assertEqual(config["hyperparameters"]["background_error_weight"], 3)
                    self.assertGreater((combination / "validation_curves.png").stat().st_size, 0)
                    previews = json.loads((combination / "checkpoint_previews.json").read_text())
                    self.assertEqual(previews[0]["sample_count"], 2)
                    self.assertGreater((combination / previews[0]["comparison_image"]).stat().st_size, 0)
                    checkpoint_path = combination / previews[0]["checkpoint"]
                    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
                    self.assertEqual(checkpoint["config"]["hyperparameters"]["background_error_weight"], 3)
                    with pipeline(name) as (model, _):
                        self.assertEqual(checkpoint["config"]["architecture"], model.architecture_metadata())
                        network = model.PairwiseCNNVAE()
                        network.load_state_dict(checkpoint["model_state_dict"], strict=True)
                    self.assertTrue(all(torch.isfinite(value).all()
                                        for value in checkpoint["model_state_dict"].values()))
                    analysis = root / name / "analysis"
                    self.run_command([TRAINING_ROOT / "analyze_runs.py", "--runs-dir", runs,
                                      "--output-dir", analysis, "--top-n", "1"], root)
                    self.assertTrue((analysis / "README.md").is_file())
                    for filename in ("top_validation_accuracy.csv", "smallest_accuracy_gap.csv"):
                        with (analysis / "smoke" / filename).open(newline="") as handle:
                            reader = csv.DictReader(handle)
                            self.assertIn("background_error_weight", reader.fieldnames)
                            for row in reader:
                                self.assertEqual(float(row["background_error_weight"]), 3)


if __name__ == "__main__":
    unittest.main()
