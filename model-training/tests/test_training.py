"""Behavioral checks for the architecture and grouped training/validation pipeline."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
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
        encoder_calls, decoder_calls, encoder_stages, decoder_stages = [], [], [], []
        decoder_inputs = []
        handles = [
            model.encoder_convs.register_forward_hook(
                lambda module, inputs, output: encoder_calls.append((id(module), tuple(output.shape)))
            ),
            model.decoder_convs.register_forward_hook(
                lambda module, inputs, output: decoder_calls.append((id(module), tuple(output.shape)))
            ),
            model.decoder_convs.register_forward_pre_hook(
                lambda module, inputs: decoder_inputs.append(inputs[0].detach().clone())
            ),
        ]
        for module in model.encoder_convs:
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(
                    lambda module, inputs, output: encoder_stages.append((id(module), tuple(output.shape)))
                ))
        for module in model.decoder_convs:
            if isinstance(module, nn.ConvTranspose2d):
                handles.append(module.register_forward_hook(
                    lambda module, inputs, output: decoder_stages.append((id(module), tuple(output.shape)))
                ))
        try:
            with torch.no_grad():
                output = model(x, x)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(encoder_calls, [(id(model.encoder_convs), (2, 128, 1, 12))] * 2)
        self.assertEqual(decoder_calls, [(id(model.decoder_convs), (2, *INPUT_SHAPE))] * 2)
        encoder_shapes = [(32, 8, 96), (64, 4, 48), (128, 2, 24), (128, 1, 12)]
        decoder_shapes = [(128, 2, 24), (64, 4, 48), (32, 8, 96), INPUT_SHAPE]
        encoder_layers = [m for m in model.encoder_convs if isinstance(m, nn.Conv2d)]
        decoder_layers = [m for m in model.decoder_convs if isinstance(m, nn.ConvTranspose2d)]
        self.assertEqual(encoder_stages,
                         [(id(m), (2, *shape)) for m, shape in zip(encoder_layers, encoder_shapes, strict=True)] * 2)
        self.assertEqual(decoder_stages,
                         [(id(m), (2, *shape)) for m, shape in zip(decoder_layers, decoder_shapes, strict=True)] * 2)
        for side in ("a", "b"):
            self.assertEqual(tuple(output[f"reconstruction_logits_{side}"].shape), (2, *INPUT_SHAPE))
            self.assertEqual(tuple(output[f"mu_{side}"].shape), (2, 32))
            self.assertEqual(tuple(output[f"log_var_{side}"].shape), (2, 32))
        for decoder_input, side in zip(decoder_inputs, ("a", "b"), strict=True):
            projected = model.decoder_projection(output[f"z_{side}"]).reshape(2, 128, 1, 12)
            torch.testing.assert_close(decoder_input, projected)
        self.assertEqual(tuple(output["rank_logits"].shape), (2,))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1_312_151)
        self.assertEqual([(m.in_channels, m.out_channels) for m in encoder_layers],
                         [(22, 32), (32, 64), (64, 128), (128, 128)])
        self.assertEqual([(m.in_channels, m.out_channels) for m in decoder_layers],
                         [(128, 128), (128, 64), (64, 32), (32, 22)])
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                self.assertEqual(module.kernel_size, (4, 4))
                self.assertEqual(module.stride, (2, 2))
                self.assertEqual(module.padding, (1, 1))
        self.assertEqual([type(m) for m in model.encoder_convs],
                         [nn.Conv2d, nn.LayerNorm, nn.ReLU] * 4)
        self.assertEqual([type(m) for m in model.decoder_convs],
                         [nn.ConvTranspose2d, nn.LayerNorm, nn.ReLU] * 3 + [nn.ConvTranspose2d])
        self.assertEqual(model.encoder_hidden[1].in_features, 1_536)
        self.assertIsInstance(model.encoder_hidden[2], nn.ReLU)
        self.assertEqual(model.decoder_projection.out_features, 1_536)
        self.assertIsInstance(model.decoder_projection, nn.Linear)
        self.assertEqual(
            [type(module) for module in model.fc_tail],
            [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear, nn.ReLU, nn.Linear],
        )
        self.assertEqual([module.p for module in model.fc_tail if isinstance(module, nn.Dropout)],
                         [0.2])
        metadata = architecture_metadata()
        self.assertEqual(metadata["architecture_version"], "model_arch_2")
        self.assertEqual(metadata["encoded_shape"], [128, 1, 12])
        self.assertEqual(metadata["flattened_features"], 1_536)
        self.assertEqual(metadata["stride"], [2, 2])
        self.assertEqual(metadata["encoder_output_shapes"], [list(shape) for shape in encoder_shapes])
        self.assertEqual(metadata["decoder_output_shapes"], [list(shape) for shape in decoder_shapes])
        self.assertEqual(metadata["encoder_convolution_activation"], "ReLU")
        self.assertEqual(metadata["encoder_hidden_activation"], "ReLU")
        self.assertEqual(metadata["decoder_convolution_activation"], "ReLU after first three transposed convolutions")
        self.assertEqual(metadata["decoder_projection_activation"], "linear")
        self.assertEqual(metadata["encoder_channels"], [22, 32, 64, 128, 128])
        self.assertEqual(metadata["decoder_channels"], [128, 128, 64, 32, 22])
        self.assertEqual(metadata["dropout_after_fc_layers"], [1])
        self.assertEqual(metadata["fc_tail_hidden_activation"], "ReLU")
        self.assertEqual(metadata["dropout"], 0.2)
        self.assertFalse(any(isinstance(m, (nn.MaxPool2d, nn.MaxUnpool2d, nn.GroupNorm, nn.LeakyReLU, nn.Sigmoid))
                             for m in model.modules()))
        normalization_layers = [m for m in model.modules() if isinstance(m, nn.LayerNorm)]
        normalization_shapes = encoder_shapes + decoder_shapes[:-1]
        self.assertEqual([m.normalized_shape for m in normalization_layers], normalization_shapes)
        for layer in normalization_layers:
            self.assertTrue(layer.elementwise_affine)
            self.assertEqual(layer.eps, 1e-5)
            self.assertEqual(tuple(layer.weight.shape), layer.normalized_shape)
            self.assertEqual(tuple(layer.bias.shape), layer.normalized_shape)
        self.assertEqual(metadata["normalization"], "LayerNorm")
        self.assertEqual(metadata["layer_norm_axes"], ["channel", "height", "width"])
        self.assertEqual(metadata["layer_norm_eps"], 1e-5)
        self.assertTrue(metadata["layer_norm_elementwise_affine"])
        self.assertEqual(metadata["encoder_layer_norm_shapes"], [list(shape) for shape in encoder_shapes])
        self.assertEqual(metadata["decoder_layer_norm_shapes"], [list(shape) for shape in decoder_shapes[:-1]])
        self.assertNotIn("group_norm_groups", metadata)
        self.assertNotIn("leaky_relu_negative_slope", metadata)

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
        hyperparameters_file = self.root / "custom-hyperparameters.txt"
        hyperparameters_file.write_text(
            "# learning_rate, lambda_rec, lambda_kl, lambda_rank\n"
            "5e-5, 0.01, 0.001, 1\n1e-6, 3, 0.0001, 0.2\n",
            encoding="utf-8",
        )
        expected_combinations = [Hyperparameters(5e-5, 0.01, 0.001, 1),
                                 Hyperparameters(1e-6, 3, 0.0001, 0.2)]
        with patch.object(training, "split_training_groups", side_effect=AssertionError("Unexpected second split")), \
                patch.object(training, "run_combination", return_value={"status": "completed"}) as run, \
                patch.object(training, "write_report"):
            training.main(["--dataset", str(self.root), "--num-training-samples", "1",
                           "--validation-fraction", "0.8", "--epochs", "1", "--device", "cpu",
                           "--hyperparameters-file", str(hyperparameters_file),
                           "--output-dir", str(output)])
        self.assertEqual(run.call_count, 2)
        self.assertEqual([call.args[3] for call in run.call_args_list], expected_combinations)
        self.assertEqual([call.args[7] for call in run.call_args_list],
                         ["combination_01", "combination_02"])
        self.assertEqual([call.args[6] for call in run.call_args_list],
                         [output.resolve() / "5e-5_1e-2_1e-3_1", output.resolve() / "1e-6_3_1e-4_2e-1"])
        self.assertTrue(all(call.args[1] == run.call_args.args[1] for call in run.call_args_list))
        self.assertEqual(len(run.call_args.args[1]), 1)
        self.assertEqual(run.call_args.args[2], [expected_validation])
        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["combination_count"], 2)
        self.assertEqual(config["hyperparameters_file"], str(hyperparameters_file.resolve()))
        self.assertEqual([Hyperparameters(**hp) for hp in config["hyperparameter_combinations"]],
                         expected_combinations)
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

    def test_every_saved_checkpoint_gets_previews_without_changing_training(self) -> None:
        expected_validation = self._save_validation_entry()
        hyperparameters_file = self.root / "preview-hyperparameters.txt"
        hyperparameters_file.write_text(
            "1e-4, 0.01, 0.001, 1\n",
            encoding="utf-8",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "with-previews"
            baseline = Path(temporary) / "without-checkpoints"
            arguments = ["--dataset", str(self.root), "--hyperparameters-file", str(hyperparameters_file),
                         "--num-training-samples", "1", "--epochs", "3", "--batch-size", "2", "--device", "cpu"]
            with patch.object(training, "plot_loss_curves"), redirect_stdout(io.StringIO()):
                training.main(arguments + ["--output-dir", str(output),
                                           "--checkpoint-start-epoch", "1", "--checkpoint-end-epoch", "2"])
                training.main(arguments + ["--output-dir", str(baseline)])
            directory_name = "1e-4_1e-2_1e-3_1"
            combination = output / directory_name
            experiments = json.loads((output / "experiments.json").read_text(encoding="utf-8"))
            self.assertEqual(experiments[0]["combination_id"], "combination_01")
            self.assertEqual(experiments[0]["directory"], directory_name)
            self.assertEqual(experiments[0]["loss_graph"], f"{directory_name}/validation_curves.png")
            previews = json.loads((combination / "checkpoint_previews.json").read_text(encoding="utf-8"))
            self.assertEqual([preview["epoch"] for preview in previews], [1, 2])
            self.assertFalse((combination / "reconstructions/epoch_003").exists())
            expected_levels = {play["level_id"] for play in expected_validation["playthroughs"]}
            manifests = []
            for preview in previews:
                self.assertEqual(preview["sample_count"], 2)
                checkpoint = torch.load(combination / preview["checkpoint"], map_location="cpu", weights_only=True)
                restored = PairwiseCNNVAE()
                restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
                self.assertEqual(checkpoint["epoch"], preview["epoch"])
                self.assertEqual(checkpoint["config"]["combination_id"], "combination_01")
                architecture = checkpoint["config"]["architecture"]
                self.assertEqual(architecture["architecture_version"], "model_arch_2")
                self.assertEqual(architecture["normalization"], "LayerNorm")
                self.assertEqual(architecture["dropout"], 0.2)
                self.assertEqual(architecture["dropout_after_fc_layers"], [1])
                manifest = json.loads((combination / preview["samples_file"]).read_text(encoding="utf-8"))
                self.assertEqual({sample["level_id"] for sample in manifest["samples"]}, expected_levels)
                self.assertTrue(all(sample["encoding"].startswith("validation_set/")
                                    for sample in manifest["samples"]))
                self.assertGreater((combination / preview["comparison_image"]).stat().st_size, 0)
                manifests.append(manifest)
            self.assertEqual([sample["level_id"] for sample in manifests[0]["samples"]],
                             [sample["level_id"] for sample in manifests[1]["samples"]])
            with_history = json.loads((combination / "train_history.json").read_text(encoding="utf-8"))
            baseline_history = json.loads((baseline / directory_name / "train_history.json").read_text(encoding="utf-8"))
            for actual, expected in zip(with_history, baseline_history, strict=True):
                for portion in ("optimization", "train", "validation"):
                    self.assertEqual(actual[portion], expected[portion])
            report = (output / "REPORT.md").read_text(encoding="utf-8")
            for preview in previews:
                self.assertIn(f"{directory_name}/{preview['comparison_image']}", report)
                self.assertIn(f"{directory_name}/{preview['checkpoint']}", report)
            self.assertFalse((baseline / directory_name / "checkpoint_previews.json").exists())

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

    def test_hyperparameter_directory_names_follow_parameter_order_and_keep_precision(self) -> None:
        cases = (
            (Hyperparameters(1e-4, 1, 1e-3, 1), "1e-4_1_1e-3_1"),
            (Hyperparameters(5e-6, 3, 0.001, 0.01), "5e-6_3_1e-3_1e-2"),
            (Hyperparameters(0.5, 0.01, 0.001, 0.2), "5e-1_1e-2_1e-3_2e-1"),
            (Hyperparameters(1.0, 3.0, 0.0, 10.25), "1_3_0_10.25"),
            (Hyperparameters(1e-10, 1, 1e-3, 1), "1e-10_1_1e-3_1"),
        )
        for hp, expected in cases:
            with self.subTest(hyperparameters=hp):
                self.assertEqual(training.hyperparameter_directory_name(hp), expected)
        first = Hyperparameters(1e-4, 0.01, 0.001, 1)
        second = Hyperparameters(1e-4, math.nextafter(0.01, math.inf), 0.001, 1)
        first_name = training.hyperparameter_directory_name(first)
        second_name = training.hyperparameter_directory_name(second)
        self.assertNotEqual(first_name, second_name)
        for hp, name in ((first, first_name), (second, second_name)):
            self.assertEqual(tuple(float(value) for value in name.split("_")),
                             (hp.learning_rate, hp.lambda_rec, hp.lambda_kl, hp.lambda_rank))

    def test_explicit_hyperparameter_rows_and_epoch_cli_defaults(self) -> None:
        hyperparameters_file = self.root / "three-combinations.txt"
        hyperparameters_file.write_text(
            "1e-6, 3, 0.0001, 0.2\n"
            "5e-5, 0.01, 0.001, 1\n"
            "1e-6, 0, 0, 0.01\n",
            encoding="utf-8",
        )
        expected = (
            Hyperparameters(1e-6, 3, 0.0001, 0.2),
            Hyperparameters(5e-5, 0.01, 0.001, 1),
            Hyperparameters(1e-6, 0, 0, 0.01),
        )
        combinations = training.load_hyperparameter_combinations(hyperparameters_file)
        self.assertEqual(len(combinations), 3)
        self.assertEqual(combinations, expected)
        self.assertEqual(training.EPOCHS, 50)
        self.assertIsNone(training.CHECKPOINT_START_EPOCH)
        self.assertIsNone(training.CHECKPOINT_END_EPOCH)
        defaults = training.parse_args([])
        self.assertEqual(defaults.hyperparameters_file, TRAINING_DIR / "hyperparameter-combinations.txt")
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

    def test_custom_hyperparameter_file_accepts_comments_and_zero_loss_weights(self) -> None:
        path = self.root / "hyperparameters.txt"
        path.write_text(
            "# learning_rate, lambda_rec, lambda_kl, lambda_rank\n\n"
            " 5e-6, 0.01, 0, 0 # ranking and KL ablation\n"
            "1e-4, 0, 0.001, 1 # reconstruction ablation\n"
            "5e-6, 0.01, 0, 1\n",
            encoding="utf-8",
        )
        expected = (Hyperparameters(5e-6, 0.01, 0, 0),
                    Hyperparameters(1e-4, 0, 0.001, 1),
                    Hyperparameters(5e-6, 0.01, 0, 1))
        self.assertEqual(training.load_hyperparameter_combinations(path), expected)
        self.assertEqual(training.parse_args(["--hyperparameters-file", str(path)]).hyperparameters_file, path)

    def test_invalid_hyperparameter_files_fail_before_loading_data_or_creating_output(self) -> None:
        valid = "1e-4, 0.01, 0.001, 1\n"
        invalid = [
            "", "# No combinations\n\n", "1e-4, 0.01, 0.001\n",
            "1e-4, 0.01, 0.001, 1, 2\n", "1e-4 0.01 0.001 1\n",
            "learning_rate, lambda_rec, lambda_kl, lambda_rank\n",
            "learning_rate: 1e-4\nlambda_kl: 0.001\nlambda_rec: 0.01\nlambda_rank: 1\n",
            valid + valid,
            valid + "0.0001, 0.010, 1e-3, 1.0\n",
        ]
        for index in range(4):
            values = ["1e-4", "0.01", "0.001", "1"]
            for invalid_value in ("-0.01", "nan", "inf", "text", ""):
                changed = values.copy()
                changed[index] = invalid_value
                invalid.append(", ".join(changed) + "\n")
        invalid.append("0, 0.01, 0.001, 1\n")
        path = self.root / "invalid-hyperparameters.txt"
        output = self.root / "should-not-exist"
        for content in invalid:
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with patch.object(training, "load_training_data") as load_data:
                    with self.assertRaisesRegex(ValueError, str(path)):
                        training.main(["--hyperparameters-file", str(path), "--output-dir", str(output)])
                load_data.assert_not_called()
                self.assertFalse(output.exists())
        with self.assertRaises(FileNotFoundError):
            training.load_hyperparameter_combinations(self.root / "missing.txt")

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
