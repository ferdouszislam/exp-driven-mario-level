"""Exercise local model training with shared data, plots, previews, and analysis."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


TRAINING_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("initial_model_architecture", "model_arch_1", "model_arch_2")
INPUT_SHAPE = (22, 16, 176)
TILE_SYMBOLS = "X S - % t T o Q @ U L 1 2 g k r K R y # |".split()


# Run in a separate interpreter so flat imports from another pipeline cannot
# satisfy this pipeline's imports during unittest discovery.
CHECK_MODEL = r'''
import json
from pathlib import Path
import sys
import torch
from torch import nn
import train
import data
import model
import plotting
import reconstruction_previews

torch.set_num_threads(2)
name, checkpoint_path = sys.argv[1:]
here = Path.cwd()
root = here.parents[1]
for module in (model, train):
    assert Path(module.__file__).resolve().parent == here, module.__file__
for module in (data, plotting, reconstruction_previews):
    assert Path(module.__file__).resolve().parent == here.parent, module.__file__
assert train.DEFAULT_DATASET == root / "datasets/fun-study-dataset-analysis/preprocessed_dataset"
assert train.DEFAULT_HYPERPARAMETERS == here / "hyperparameter-combinations.txt"
assert reconstruction_previews.RENDERER_PATH.is_file()

expected_count = {"initial_model_architecture": 1117591,
                  "model_arch_1": 533175, "model_arch_2": 1276951}[name]
net = model.PairwiseCNNVAE().eval()
metadata = model.architecture_metadata()
assert metadata["architecture_version"] == name + "_crop16"
assert sum(p.numel() for p in net.parameters()) == expected_count
expected_depth = 3 if name == "model_arch_1" else 4
assert sum(isinstance(m, nn.Conv2d) for m in net.modules()) == expected_depth
assert sum(isinstance(m, nn.ConvTranspose2d) for m in net.modules()) == expected_depth
expected_dropout = [0.1] if name == "initial_model_architecture" else (
    [0.2, 0.2] if name == "model_arch_1" else [0.2])
assert [m.p for m in net.fc_tail if isinstance(m, nn.Dropout)] == expected_dropout
if name == "model_arch_1":
    assert isinstance(net.decoder_projection, nn.Sequential)
    assert isinstance(net.decoder_projection[1], nn.ReLU)
else:
    assert isinstance(net.decoder_projection, nn.Linear)
if name != "initial_model_architecture":
    assert not any(isinstance(m, (nn.GroupNorm, nn.LeakyReLU)) for m in net.modules())
    expected_encoder = [(32, 8, 88), (64, 4, 44)] + (
        [(32, 2, 22)] if name == "model_arch_1" else [(128, 2, 22), (128, 1, 11)])
    expected_decoder = ([(64, 4, 44), (32, 8, 88)] if name == "model_arch_1" else
                        [(128, 2, 22), (64, 4, 44), (32, 8, 88)])
    norms = [m for m in net.modules() if isinstance(m, nn.LayerNorm)]
    assert [tuple(m.normalized_shape) for m in norms] == expected_encoder + expected_decoder
    assert all(m.eps == 1e-5 and m.elementwise_affine for m in norms)
    assert metadata["normalization"] == "LayerNorm"
    assert metadata["dropout_after_fc_layers"] == ([1, 2] if name == "model_arch_1" else [1])

x_a, x_b = torch.rand(2, 22, 16, 176), torch.rand(2, 22, 16, 176)
output = net(x_a, x_b)
for side in ("a", "b"):
    assert output["reconstruction_logits_" + side].shape == x_a.shape
    assert output["mu_" + side].shape == (2, 32)
assert output["rank_logits"].shape == (2,)
output["rank_logits"].sum().backward()
assert net.encoder_convs[0].weight.grad.abs().sum() > 0
assert all(p.grad is None for p in net.decoder_projection.parameters())

checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
assert checkpoint["config"]["architecture"] == metadata
assert checkpoint["config"]["parameter_count"] == expected_count
net.load_state_dict(checkpoint["model_state_dict"], strict=True)
assert checkpoint["epoch"] == 1
assert all(torch.isfinite(p).all() for p in net.parameters())
'''


def write_dataset(root: Path) -> None:
    """One training pair and one independent saved validation pair."""
    root.mkdir()
    channels = {
        "shape": list(INPUT_SHAPE), "dtype": "uint8",
        "axis_order": ["channel", "row", "column"], "path_channel_index": 21,
        "channels": [{"index": index, "symbol": symbol,
                      "name": "PATH" if index == 21 else symbol}
                     for index, symbol in enumerate(TILE_SYMBOLS + ["P"])],
    }
    (root / "channels.json").write_text(json.dumps(channels), encoding="utf-8")
    entries = []
    for split in ("train_set", "validation_set"):
        instance = f"{split}-pair"
        relative = f"{split}/{instance}"
        directory = root / relative
        directory.mkdir(parents=True)
        levels = [f"{split}-winner", f"{split}-other"]
        (directory / "ranking.txt").write_text("\n".join(levels) + "\n", encoding="utf-8")
        plays = []
        for side, level in enumerate(levels):
            array = np.zeros(INPUT_SHAPE, dtype=np.uint8)
            array[2] = 1
            array[21, 5 + side, 10:15] = 1
            np.save(directory / f"{level}.npy", array, allow_pickle=False)
            plays.append({"level_id": level, "playthrough_key": f"participant_{level}",
                          "files": {"encoding": f"{relative}/{level}.npy"}})
        entries.append({
            "instance_id": instance, "split": split, "split_group_id": instance,
            "relative_directory": relative, "ranking_file": f"{relative}/ranking.txt",
            "preferred_level_id": levels[0], "less_preferred_level_id": levels[1],
            "playthroughs": plays,
        })
    # An unreadable held-out entry also checks that no test data is required.
    entries.append({"split": "test_set", "instance_id": "held-out"})
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": 2, "instances": entries}), encoding="utf-8")


class ArchitecturePipelineTests(unittest.TestCase):
    def run_command(self, arguments: list[str], cwd: Path) -> None:
        result = subprocess.run([sys.executable, *arguments], cwd=cwd,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shared_utilities_import_without_selecting_an_architecture(self) -> None:
        self.run_command(["-c", '''
import sys
import data
import plotting
import reconstruction_previews
import analyze_runs

assert "model" not in sys.modules
assert "train" not in sys.modules
assert data.INPUT_SHAPE == (22, 16, 176)
assert reconstruction_previews.RENDERER_PATH.is_file()
'''], TRAINING_ROOT)

    def test_each_folder_trains_restores_and_analyzes_its_own_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            write_dataset(dataset)
            hyperparameters = root / "one-combination.txt"
            hyperparameters.write_text("1e-4, 1, 1e-3, 1e-2\n", encoding="utf-8")
            for name in ARCHITECTURES:
                with self.subTest(architecture=name):
                    folder = TRAINING_ROOT / name
                    runs = root / name / "runs"
                    output = runs / "smoke"
                    self.run_command([
                        "train.py", "--dataset", str(dataset),
                        "--hyperparameters-file", str(hyperparameters),
                        "--epochs", "1", "--batch-size", "2", "--cpu-threads", "2",
                        "--device", "cpu", "--checkpoint-start-epoch", "1",
                        "--checkpoint-end-epoch", "1", "--output-dir", str(output),
                    ], folder)
                    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
                    self.assertEqual(config["architecture"]["architecture_version"], name + "_crop16")
                    self.assertEqual(config["fitting_comparisons"], 1)
                    self.assertEqual(config["validation_comparisons"], 1)
                    experiment = json.loads((output / "experiments.json").read_text(encoding="utf-8"))[0]
                    self.assertEqual(experiment["status"], "completed")
                    combination = output / experiment["directory"]
                    self.assertGreater((combination / "validation_curves.png").stat().st_size, 0)
                    preview = json.loads((combination / "checkpoint_previews.json").read_text(encoding="utf-8"))[0]
                    self.assertEqual(preview["sample_count"], 2)
                    self.assertGreater((combination / preview["comparison_image"]).stat().st_size, 0)
                    self.run_command(["-c", CHECK_MODEL, name,
                                      str(combination / preview["checkpoint"])], folder)
                    analysis = root / name / "analysis"
                    self.run_command(["../analyze_runs.py", "--runs-dir", str(runs),
                                      "--output-dir", str(analysis), "--top-n", "1"], folder)
                    self.assertTrue((analysis / "README.md").is_file())
                    self.assertTrue((analysis / "smoke/top_validation_accuracy.csv").is_file())
                    self.assertTrue((analysis / "smoke/smallest_accuracy_gap.csv").is_file())


if __name__ == "__main__":
    unittest.main()
