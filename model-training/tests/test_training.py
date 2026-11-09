"""Behavioral checks for the architecture and grouped training/validation pipeline."""

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
from model import PairwiseCNNVAE, architecture_metadata
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

    def _save_validation_entry(self) -> dict:
        """Move one independent fixture group into a saved validation split."""
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["instances"][3]
        old_directory = self.root / entry["relative_directory"]
        entry["split"] = "validation_set"
        entry["relative_directory"] = f"validation_set/{entry['instance_id']}"
        entry["ranking_file"] = f"{entry['relative_directory']}/ranking.txt"
        for play in entry["playthroughs"]:
            play["files"]["encoding"] = f"{entry['relative_directory']}/{play['level_id']}.npy"
        new_directory = self.root / entry["relative_directory"]
        new_directory.parent.mkdir()
        old_directory.rename(new_directory)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return entry

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
        self.assertEqual(encoder_calls, [(id(model.encoder_convs), (2, 32, 2, 24))] * 2)
        self.assertEqual(decoder_calls, [(id(model.decoder_convs), (2, *INPUT_SHAPE))] * 2)
        self.assertEqual(tuple(model.encoder_convs[0](x).shape), (2, 32, 8, 96))
        self.assertEqual(tuple(model.decoder_convs[0](torch.zeros(2, 32, 2, 24)).shape), (2, 64, 4, 48))
        for side in ("a", "b"):
            self.assertEqual(tuple(output[f"reconstruction_logits_{side}"].shape), (2, *INPUT_SHAPE))
            self.assertEqual(tuple(output[f"mu_{side}"].shape), (2, 32))
            self.assertEqual(tuple(output[f"log_var_{side}"].shape), (2, 32))
        self.assertEqual(tuple(output["rank_logits"].shape), (2,))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 566_327)
        self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in model.modules()), 3)
        self.assertEqual(sum(isinstance(m, nn.ConvTranspose2d) for m in model.modules()), 3)
        self.assertEqual([(m.in_channels, m.out_channels) for m in model.modules() if isinstance(m, nn.Conv2d)],
                         [(22, 32), (32, 64), (64, 32)])
        self.assertEqual([(m.in_channels, m.out_channels) for m in model.modules() if isinstance(m, nn.ConvTranspose2d)],
                         [(32, 64), (64, 32), (32, 22)])
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                self.assertEqual(module.kernel_size, (4, 4))
                self.assertEqual(module.stride, (2, 2))
                self.assertEqual(module.padding, (1, 1))
        self.assertIsInstance(model.encoder_convs[2], nn.ReLU)
        self.assertIsInstance(model.encoder_convs[5], nn.ReLU)
        self.assertIsInstance(model.encoder_convs[8], nn.ReLU)
        self.assertIsInstance(model.decoder_convs[2], nn.ReLU)
        self.assertIsInstance(model.decoder_convs[5], nn.ReLU)
        self.assertFalse(any(isinstance(module, nn.LeakyReLU) for module in model.modules()))
        self.assertEqual(model.encoder_hidden[1].in_features, 1_536)
        self.assertIsInstance(model.encoder_hidden[2], nn.ReLU)
        self.assertEqual(model.decoder_projection.out_features, 1_536)
        self.assertIsInstance(model.decoder_activation, nn.ReLU)
        self.assertEqual(
            [type(module) for module in model.fc_tail],
            [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.ReLU, nn.Dropout, nn.Linear],
        )
        self.assertEqual([module.p for module in model.fc_tail if isinstance(module, nn.Dropout)],
                         [0.2, 0.2])
        metadata = architecture_metadata()
        self.assertEqual(metadata["architecture_version"], "model_arch_1")
        self.assertEqual(metadata["encoded_shape"], [32, 2, 24])
        self.assertEqual(metadata["flattened_features"], 1_536)
        self.assertEqual(metadata["stride"], [2, 2])
        self.assertEqual(metadata["encoder_output_shapes"], [[32, 8, 96], [64, 4, 48], [32, 2, 24]])
        self.assertEqual(metadata["decoder_output_shapes"], [[64, 4, 48], [32, 8, 96], [22, 16, 192]])
        self.assertEqual(metadata["encoder_convolution_activation"], "ReLU")
        self.assertEqual(metadata["decoder_convolution_activation"], "ReLU after first two transposed convolutions")
        self.assertEqual(metadata["encoder_channels"], [22, 32, 64, 32])
        self.assertEqual(metadata["decoder_channels"], [32, 64, 32, 22])
        self.assertEqual(metadata["dropout_after_fc_layers"], [1, 2])
        self.assertFalse(any(isinstance(m, (nn.MaxPool2d, nn.MaxUnpool2d)) for m in model.modules()))
        self.assertFalse(any(isinstance(m, nn.GroupNorm) for m in model.modules()))
        normalization_shapes = [(32, 8, 96), (64, 4, 48), (32, 2, 24), (64, 4, 48), (32, 8, 96)]
        normalization_layers = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
        self.assertEqual([m.normalized_shape for m in normalization_layers], normalization_shapes)
        for layer in normalization_layers:
            self.assertTrue(layer.elementwise_affine)
            self.assertEqual(layer.eps, 1e-5)
            self.assertEqual(tuple(layer.weight.shape), layer.normalized_shape)
            self.assertEqual(tuple(layer.bias.shape), layer.normalized_shape)
        self.assertEqual(metadata["normalization"], "LayerNorm")
        self.assertEqual(metadata["layer_norm_axes"], ["channel", "height", "width"])
        self.assertEqual(metadata["encoder_layer_norm_shapes"], [[32, 8, 96], [64, 4, 48], [32, 2, 24]])
        self.assertEqual(metadata["decoder_layer_norm_shapes"], [[64, 4, 48], [32, 8, 96]])
        self.assertNotIn("group_norm_groups", metadata)

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
             [[2, 1], [2, 3]], 5 / 8, 2 / 3),
            ([0, 1], [1, 1], [[0, 1], [0, 1]], 0.5, 2 / 3),
            ([0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1],
             [[4, 0], [1, 3]], 7 / 8, 6 / 7),
            ([0, 0], [0, 0], [[2, 0], [0, 0]], 1.0, 0.0),
        ]
        for targets, predictions, matrix, accuracy, positive_f1 in cases:
            with self.subTest(predictions=predictions):
                metrics = classification_metrics(targets, predictions)
                self.assertEqual(metrics["confusion_matrix"], matrix)
                self.assertAlmostEqual(metrics["accuracy"], accuracy)
                self.assertAlmostEqual(metrics["f1_score"], positive_f1)
                self.assertEqual(metrics["f1_positive_label"], 1)
                self.assertNotIn("f1_by_class", metrics)
                self.assertNotIn("accuracy_by_class", metrics)
                self.assertNotIn("macro_f1", metrics)
        with self.assertRaises(ValueError):
            classification_metrics([], [])
        losses = {name: 0.25 for name in training.LOSS_NAMES}
        measured = {**losses, **classification_metrics([0, 1], [0, 1]),
                    "normalized_hamming_distance": 0.25}
        row = {"epoch": 1, "seconds": 0.1, "optimization": losses,
               "train": measured, "validation": measured}
        expected_fields = {"epoch", "seconds"}
        expected_fields.update(f"optimization_{name}" for name in training.LOSS_NAMES)
        for portion in ("train", "validation"):
            expected_fields.update(
                f"{portion}_{name}" for name in (
                    *training.LOSS_NAMES, "accuracy", "f1_score", "normalized_hamming_distance"
                )
            )
        self.assertEqual(set(training.flatten_epoch_metrics(row)), expected_fields)

    def test_normalized_hamming_counts_tile_positions_across_uneven_batches(self) -> None:
        data = load_training_data(self.root, None, 42)
        model = PairwiseCNNVAE()
        hp = Hyperparameters(1e-3, 1.0, 0.001, 1.0)
        cases = (("all_match", 0.0), ("four_mismatches", 4 / (2 * 4 * 16 * 192)),
                 ("all_different", 1.0))
        for scenario, expected_distance in cases:
            with self.subTest(scenario=scenario):
                batch_index = 0

                def controlled_output(x_a, x_b):
                    nonlocal batch_index
                    predicted_tile = 0 if scenario == "all_different" else 2
                    reconstruction_a = torch.full_like(x_a, -10.0)
                    reconstruction_b = torch.full_like(x_b, -10.0)
                    for reconstruction in (reconstruction_a, reconstruction_b):
                        reconstruction[:, predicted_tile] = 10.0
                        # The path logit dominates every tile logit. Geometry
                        # comparison must still select among channels 0..20.
                        reconstruction[:, 21] = 1000.0
                    if scenario == "four_mismatches":
                        if batch_index == 0:
                            reconstruction_a[0, 0, 0, 0] = 20.0
                            reconstruction_b[2, 1, 1, 2] = 20.0
                        else:
                            reconstruction_a[0, 0, 2, 3] = 20.0
                            reconstruction_b[0, 1, 3, 4] = 20.0
                    batch_index += 1
                    latent = torch.zeros(x_a.shape[0], 32)
                    return {
                        "reconstruction_logits_a": reconstruction_a,
                        "reconstruction_logits_b": reconstruction_b,
                        "mu_a": latent,
                        "mu_b": latent,
                        "log_var_a": latent,
                        "log_var_b": latent,
                        "z_a": latent,
                        "z_b": latent,
                        "rank_logits": torch.zeros(x_a.shape[0]),
                    }

                with patch.object(model, "forward", side_effect=controlled_output) as forward, \
                        patch.object(model, "preference_logits",
                                     side_effect=lambda z_a, z_b: torch.zeros(z_a.shape[0])):
                    metrics, _ = evaluate_training_pairs(
                        model, data, data.entries, hp, 3, torch.device("cpu"), 123
                    )
                self.assertEqual(forward.call_count, 2)
                self.assertAlmostEqual(metrics["normalized_hamming_distance"], expected_distance, places=12)
                self.assertGreaterEqual(metrics["normalized_hamming_distance"], 0.0)
                self.assertLessEqual(metrics["normalized_hamming_distance"], 1.0)

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

    def test_saved_validation_is_loaded_whole_alongside_capped_training(self) -> None:
        expected_validation = self._save_validation_entry()
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if "test_set" in path.parts:
                raise AssertionError(f"Attempted test access: {path}")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            data = load_training_data(self.root, 1, 42)
        self.assertEqual(data.available_comparisons, 3)
        self.assertEqual(len(data.entries), 1)
        self.assertEqual(data.validation_entries, [expected_validation])
        validation = ComparisonDataset(data, data.validation_entries, ordered=True)
        self.assertEqual(len(validation), 2)
        self.assertEqual([validation[i][2].item() for i in range(2)], [1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "only selected"):
            ComparisonDataset(data, [{"split": "test_set", "instance_id": "test"}], ordered=False)

        encoding = self.root / expected_validation["playthroughs"][0]["files"]["encoding"]
        array = np.load(encoding, allow_pickle=False)
        array[21, 0, 0] = 2
        np.save(encoding, array, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "binary multi-hot"):
            load_training_data(self.root, 1, 42)

    def test_saved_validation_rejects_shared_groups_and_playthroughs(self) -> None:
        self._save_validation_entry()
        manifest_path = self.root / "manifest.json"
        original = manifest_path.read_text(encoding="utf-8")
        for identity in ("group", "playthrough"):
            with self.subTest(identity=identity):
                manifest = json.loads(original)
                train, validation = manifest["instances"][0], manifest["instances"][3]
                if identity == "group":
                    validation["split_group_id"] = train["split_group_id"]
                else:
                    validation["playthroughs"][0]["playthrough_key"] = train["playthroughs"][0]["playthrough_key"]
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "share groups or playthroughs"):
                    load_training_data(self.root, 1, 42)

    def test_main_uses_saved_validation_without_another_split(self) -> None:
        expected_validation = self._save_validation_entry()
        output_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(output_temporary.cleanup)
        output = Path(output_temporary.name) / "run"
        with patch.object(training, "split_training_groups", side_effect=AssertionError("Unexpected second split")), \
                patch.object(training, "HYPERPARAMETER_COMBINATIONS", training.HYPERPARAMETER_COMBINATIONS[:1]), \
                patch.object(training, "run_combination", return_value={"status": "completed"}) as run, \
                patch.object(training, "write_report"):
            training.main(["--dataset", str(self.root), "--num-training-samples", "1",
                           "--validation-fraction", "0.8", "--epochs", "1", "--device", "cpu",
                           "--output-dir", str(output)])
        self.assertEqual(len(run.call_args.args[1]), 1)
        self.assertEqual(run.call_args.args[2], [expected_validation])
        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["validation_source"], "validation_set")
        self.assertIsNone(config["validation_fraction"])
        split = json.loads((output / "split.json").read_text(encoding="utf-8"))
        self.assertEqual(split["validation_instance_ids"], [expected_validation["instance_id"]])
        self.assertEqual(json.loads((output / "selected_validation_comparisons.json").read_text(encoding="utf-8")),
                         [expected_validation])

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

    def test_sixteen_configuration_grid_and_epoch_cli_defaults(self) -> None:
        expected = tuple(
            Hyperparameters(learning_rate, reconstruction, kl, ranking)
            for learning_rate in (1e-4, 1e-5)
            for kl in (1e-3, 1e-4)
            for reconstruction, ranking in ((0.5, 0.5), (1.0, 0.5), (0.5, 1.0), (1.0, 1.0))
        )
        self.assertEqual(len(training.HYPERPARAMETER_COMBINATIONS), 16)
        self.assertEqual(len(set(training.HYPERPARAMETER_COMBINATIONS)), 16)
        self.assertEqual(training.HYPERPARAMETER_COMBINATIONS, expected)
        self.assertEqual(training.EPOCHS, 50)
        self.assertIsNone(training.CHECKPOINT_START_EPOCH)
        self.assertIsNone(training.CHECKPOINT_END_EPOCH)
        defaults = training.parse_args([])
        self.assertEqual(defaults.epochs, 50)
        self.assertEqual((defaults.checkpoint_start_epoch, defaults.checkpoint_end_epoch), (None, None))
        self.assertEqual(training.parse_args(["--epochs", "100"]).epochs, 100)
        self.assertFalse(hasattr(training.parse_args([]), "tuning_epochs"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected:
            training.parse_args(["--tuning-epochs", "2"])
        self.assertEqual(rejected.exception.code, 2)
        valid_ranges = [
            (["--epochs", "2"], (None, None)),
            (["--epochs", "5", "--checkpoint-start-epoch", "2", "--checkpoint-end-epoch", "3"], (2, 3)),
            (["--epochs", "5", "--checkpoint-start-epoch", "2", "--checkpoint-end-epoch", "2"], (2, 2)),
            (["--epochs", "5", "--checkpoint-start-epoch", "10", "--checkpoint-end-epoch", "12"], (10, 12)),
        ]
        for arguments, expected_range in valid_ranges:
            with self.subTest(arguments=arguments):
                parsed = training.parse_args(arguments)
                self.assertEqual((parsed.checkpoint_start_epoch, parsed.checkpoint_end_epoch), expected_range)
        invalid_ranges = [
            ["--checkpoint-start-epoch", "2"],
            ["--checkpoint-end-epoch", "3"],
            ["--checkpoint-start-epoch", "0", "--checkpoint-end-epoch", "3"],
            ["--checkpoint-start-epoch", "2", "--checkpoint-end-epoch", "0"],
            ["--checkpoint-start-epoch", "-1", "--checkpoint-end-epoch", "3"],
            ["--checkpoint-start-epoch", "2", "--checkpoint-end-epoch", "-1"],
            ["--checkpoint-start-epoch", "3", "--checkpoint-end-epoch", "2"],
        ]
        for arguments in invalid_ranges:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as rejected:
                    training.parse_args(arguments)
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
