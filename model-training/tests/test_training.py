"""Behavioral checks for the specified architecture and train-only pipeline."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn


TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

from data import ComparisonDataset, INPUT_SHAPE, TILE_SYMBOLS, load_training_data
from model import PairwiseCNNVAE
import train as training
from train import (
    Hyperparameters,
    classification_metrics,
    evaluate_training_pairs,
    loss_components,
    make_loader,
    split_training_groups,
    weighted_loss,
)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(4)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source_entries = self._write_fixture()

    def _write_fixture(self) -> list[dict]:
        """Four comparisons in three indivisible groups plus an unreadable test entry."""
        channels = {
            "shape": list(INPUT_SHAPE),
            "dtype": "uint8",
            "axis_order": ["channel", "row", "column"],
            "path_channel_index": 21,
            "channels": [
                {"index": i, "symbol": symbol, "name": "PATH" if i == 21 else symbol}
                for i, symbol in enumerate(TILE_SYMBOLS + ["P"])
            ],
        }
        (self.root / "channels.json").write_text(json.dumps(channels), encoding="utf-8")
        entries = []
        for index, group in enumerate(("shared", "shared", "solo-b", "solo-c")):
            instance = f"comparison-{index}"
            preferred = "shared-winner" if group == "shared" else f"winner-{index}"
            other = f"other-{index}"
            relative = f"train_set/{instance}"
            directory = self.root / relative
            directory.mkdir(parents=True)
            (directory / "ranking.txt").write_text(f"{preferred}\n{other}\n", encoding="utf-8")
            plays = []
            for side, level in enumerate((preferred, other)):
                array = np.zeros(INPUT_SHAPE, dtype=np.uint8)
                array[2] = 1
                array[21, 3 + side, index] = 1
                np.save(directory / f"{level}.npy", array, allow_pickle=False)
                plays.append({
                    "level_id": level,
                    "playthrough_key": f"participant-{group}_{level}",
                    "files": {"encoding": f"{relative}/{level}.npy"},
                })
            entries.append({
                "instance_id": instance,
                "split": "train_set",
                "split_group_id": group,
                "relative_directory": relative,
                "ranking_file": f"{relative}/ranking.txt",
                "preferred_level_id": preferred,
                "less_preferred_level_id": other,
                "playthroughs": plays,
            })
        # Its incomplete metadata would fail immediately if the training loader
        # tried to process test entries. There are deliberately no test files.
        test_entry = {"split": "test_set", "instance_id": "never-read-this-test-entry"}
        (self.root / "manifest.json").write_text(
            json.dumps({"schema_version": 2, "instances": entries + [test_entry]}),
            encoding="utf-8",
        )
        return entries

    def test_architecture_shapes_and_one_shared_parameter_set(self) -> None:
        model = PairwiseCNNVAE()
        x = torch.zeros(2, *INPUT_SHAPE)
        encoder_calls, decoder_calls = [], []
        encoder_handle = model.encoder_convs.register_forward_hook(
            lambda module, inputs, output: encoder_calls.append((id(module), tuple(output.shape)))
        )
        decoder_handle = model.decoder_convs.register_forward_hook(
            lambda module, inputs, output: decoder_calls.append((id(module), tuple(output.shape)))
        )
        try:
            output = model(x, x)
        finally:
            encoder_handle.remove()
            decoder_handle.remove()
        self.assertEqual(encoder_calls, [(id(model.encoder_convs), (2, 128, 1, 12))] * 2)
        self.assertEqual(decoder_calls, [(id(model.decoder_convs), (2, *INPUT_SHAPE))] * 2)
        for side in ("a", "b"):
            self.assertEqual(tuple(output[f"reconstruction_logits_{side}"].shape), (2, *INPUT_SHAPE))
            self.assertEqual(tuple(output[f"mu_{side}"].shape), (2, 32))
            self.assertEqual(tuple(output[f"log_var_{side}"].shape), (2, 32))
        self.assertEqual(tuple(output["rank_logits"].shape), (2,))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1_138_199)
        self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in model.modules()), 4)
        self.assertEqual(sum(isinstance(m, nn.ConvTranspose2d) for m in model.modules()), 4)
        self.assertFalse(any(isinstance(m, (nn.MaxPool2d, nn.MaxUnpool2d)) for m in model.modules()))
        self.assertTrue(all(m.num_groups == 8 for m in model.modules() if isinstance(m, nn.GroupNorm)))

    def test_loss_matches_hand_calculated_bce_kl_and_weighting(self) -> None:
        x = torch.zeros(2, *INPUT_SHAPE)
        output = {
            "reconstruction_logits_a": torch.zeros_like(x),
            "reconstruction_logits_b": torch.zeros_like(x),
            "rank_logits": torch.zeros(2),
            "mu_a": torch.ones(2, 32),
            "mu_b": torch.ones(2, 32),
            "log_var_a": torch.zeros(2, 32),
            "log_var_b": torch.zeros(2, 32),
        }
        parts = loss_components(output, x, x, torch.tensor([0.0, 1.0]))
        self.assertAlmostEqual(parts["reconstruction_bce"].item(), math.log(2), places=6)
        self.assertAlmostEqual(parts["ranking_bce"].item(), math.log(2), places=6)
        self.assertAlmostEqual(parts["kl_divergence"].item(), 16.0, places=6)
        hp = Hyperparameters(1e-3, 2.0, 0.5, 3.0)
        self.assertAlmostEqual(weighted_loss(parts, hp).item(), 5 * math.log(2) + 8, places=5)

    def test_ranking_gradients_skip_decoder_and_joint_loss_reaches_all_parts(self) -> None:
        torch.manual_seed(4)
        model = PairwiseCNNVAE()
        x_a, x_b = torch.rand(2, *INPUT_SHAPE), torch.rand(2, *INPUT_SHAPE)
        target = torch.tensor([1.0, 0.0])
        output = model(x_a, x_b)
        ranking = loss_components(output, x_a, x_b, target)["ranking_bce"]
        ranking.backward()
        self.assertGreater(model.encoder_convs[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.mu_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.log_var_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.fc_tail[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.decoder_convs.parameters()))
        self.assertTrue(all(p.grad is None for p in model.decoder_projection.parameters()))
        model.zero_grad(set_to_none=True)
        parts = loss_components(model(x_a, x_b), x_a, x_b, target)
        weighted_loss(parts, Hyperparameters(1e-3, 1.0, 0.01, 1.0)).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertGreater(model.decoder_projection.weight.grad.abs().sum().item(), 0)

    def test_reporting_reuses_samples_in_reverse_order_and_is_reproducible(self) -> None:
        data = load_training_data(self.root, None, 42)
        model = PairwiseCNNVAE()
        hp = Hyperparameters(1e-3, 1.0, 0.001, 1.0)
        with patch.object(model, "preference_logits", wraps=model.preference_logits) as preference:
            metrics, rows = evaluate_training_pairs(model, data, data.entries, hp, 2, torch.device("cpu"), 123)
        calls = preference.call_args_list
        self.assertEqual(len(calls), 4)
        for index in (0, 2):
            first_a, first_b = calls[index].args
            reverse_a, reverse_b = calls[index + 1].args
            self.assertIs(first_a, reverse_b)
            self.assertIs(first_b, reverse_a)
        metrics_again, rows_again = evaluate_training_pairs(
            model, data, data.entries, hp, 2, torch.device("cpu"), 123
        )
        self.assertEqual(metrics, metrics_again)
        self.assertEqual(rows, rows_again)
        self.assertEqual(metrics["class_support"], {"0": 4, "1": 4})
        self.assertEqual(metrics["ordered_examples"], 8)
        self.assertFalse(model.training)
        self.assertFalse(metrics["dropout_enabled"])

    def test_accuracy_binary_f1_and_single_predicted_class(self) -> None:
        cases = [
            ([0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 0, 0, 1, 1, 1],
             [[2, 1], [2, 3]], 5 / 8, 2 / 3, 4 / 7, {"0": 2 / 3, "1": 3 / 5}),
            ([0, 1], [1, 1], [[0, 1], [0, 1]], 0.5, 2 / 3, 0.0, {"0": 0.0, "1": 1.0}),
            ([0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1],
             [[4, 0], [1, 3]], 7 / 8, 6 / 7, 8 / 9, {"0": 1.0, "1": 3 / 4}),
            ([0, 0], [0, 0], [[2, 0], [0, 0]], 1.0, 0.0, 1.0, {"0": 1.0, "1": 0.0}),
        ]
        for targets, predictions, matrix, accuracy, positive_f1, negative_f1, class_accuracy in cases:
            with self.subTest(predictions=predictions):
                metrics = classification_metrics(targets, predictions)
                self.assertEqual(metrics["confusion_matrix"], matrix)
                self.assertAlmostEqual(metrics["accuracy"], accuracy)
                self.assertAlmostEqual(metrics["f1_score"], positive_f1)
                self.assertEqual(metrics["f1_positive_label"], 1)
                self.assertAlmostEqual(metrics["f1_by_class"]["0"], negative_f1)
                self.assertAlmostEqual(metrics["f1_by_class"]["1"], positive_f1)
                self.assertEqual(metrics["accuracy_by_class"], class_accuracy)
                self.assertNotIn("macro_f1", metrics)
        with self.assertRaises(ValueError):
            classification_metrics([], [])

    def test_grouped_cap_and_validation_keep_reused_playthroughs_together(self) -> None:
        data = load_training_data(self.root, 3, 42)
        again = load_training_data(self.root, 3, 42)
        self.assertLessEqual(len(data.entries), 3)
        self.assertEqual(data.entries, again.entries)
        self.assertEqual(data.dataset_fingerprint, again.dataset_fingerprint)
        for group in {entry["split_group_id"] for entry in data.entries}:
            expected = {entry["instance_id"] for entry in self.source_entries if entry["split_group_id"] == group}
            actual = {entry["instance_id"] for entry in data.entries if entry["split_group_id"] == group}
            self.assertEqual(actual, expected)
        fitting, validation = split_training_groups(data.entries, 0.2, 42)
        self.assertTrue(fitting)
        self.assertTrue(validation)
        self.assertFalse({e["split_group_id"] for e in fitting} & {e["split_group_id"] for e in validation})
        fitting_plays = {p["playthrough_key"] for e in fitting for p in e["playthroughs"]}
        validation_plays = {p["playthrough_key"] for e in validation for p in e["playthroughs"]}
        self.assertFalse(fitting_plays & validation_plays)
        self.assertEqual(len(fitting) + len(validation), len(data.entries))

    def test_loader_never_opens_test_paths(self) -> None:
        original_open = Path.open
        opened = []

        def guarded_open(path, *args, **kwargs):
            if "test_set" in path.parts:
                raise AssertionError(f"Attempted test access: {path}")
            opened.append(path)
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            data = load_training_data(self.root, None, 42)
        self.assertEqual(data.available_comparisons, 4)
        self.assertEqual(len(data.entries), 4)
        self.assertTrue(any(path.suffix == ".npy" for path in opened))

    def test_ordered_dataset_is_balanced_and_seeded_loader_is_deterministic(self) -> None:
        data = load_training_data(self.root, None, 42)
        dataset = ComparisonDataset(data, data.entries, ordered=True)
        self.assertEqual(len(dataset), 8)
        for index in range(0, len(dataset), 2):
            preferred, other, label = dataset[index]
            reverse_first, reverse_second, reverse_label = dataset[index + 1]
            self.assertIs(preferred, reverse_second)
            self.assertIs(other, reverse_first)
            self.assertEqual((label.item(), reverse_label.item()), (1.0, 0.0))
        first = list(make_loader(data, data.entries, 3, ordered=True, shuffle=True, seed=7))
        second = list(make_loader(data, data.entries, 3, ordered=True, shuffle=True, seed=7))
        for batch_a, batch_b in zip(first, second, strict=True):
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(batch_a, batch_b, strict=True)))

    def test_loader_rejects_nonbinary_encoding_and_inconsistent_rank(self) -> None:
        entry = self.source_entries[0]
        array_path = self.root / entry["playthroughs"][0]["files"]["encoding"]
        array = np.load(array_path, allow_pickle=False)
        array[21, 0, 0] = 2
        np.save(array_path, array, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "binary multi-hot"):
            load_training_data(self.root, None, 42)
        array[21, 0, 0] = 0
        np.save(array_path, array, allow_pickle=False)
        (self.root / entry["ranking_file"]).write_text("wrong\norder\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Ranking"):
            load_training_data(self.root, None, 42)

    def test_twelve_configuration_grid_and_epoch_cli_defaults(self) -> None:
        expected = {
            Hyperparameters(learning_rate, reconstruction, kl, ranking)
            for learning_rate in (1e-3, 3e-4)
            for kl in (1e-3, 1e-2)
            for reconstruction, ranking in ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
        }
        self.assertEqual(len(training.HYPERPARAMETER_COMBINATIONS), 12)
        self.assertEqual(set(training.HYPERPARAMETER_COMBINATIONS), expected)
        self.assertEqual(training.EPOCHS, 100)
        self.assertEqual(training.CHECKPOINT_START_EPOCH, 21)
        self.assertEqual(training.parse_args([]).epochs, 100)
        self.assertEqual(training.parse_args(["--epochs", "100"]).epochs, 100)
        self.assertFalse(hasattr(training.parse_args([]), "tuning_epochs"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected:
            training.parse_args(["--tuning-epochs", "2"])
        self.assertEqual(rejected.exception.code, 2)

    def test_evaluation_preserves_training_random_number_states(self) -> None:
        data = load_training_data(self.root, None, 42)
        model = PairwiseCNNVAE()
        hp = Hyperparameters(1e-3, 1.0, 0.001, 1.0)
        random.seed(901)
        np.random.seed(902)
        torch.manual_seed(903)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        evaluate_training_pairs(model, data, data.entries, hp, 2, torch.device("cpu"), 123)
        self.assertEqual(random.getstate(), python_state)
        restored_numpy_state = np.random.get_state()
        self.assertEqual(restored_numpy_state[0], numpy_state[0])
        np.testing.assert_array_equal(restored_numpy_state[1], numpy_state[1])
        self.assertEqual(restored_numpy_state[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))


if __name__ == "__main__":
    unittest.main()
