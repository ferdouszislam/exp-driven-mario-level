"""Check epoch selection, source fidelity, and generated ranking artifacts."""

from __future__ import annotations

from contextlib import redirect_stdout
import csv
from decimal import Decimal
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


TRAINING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_DIR))

import analyze_runs as analysis


class AnalyzeRunsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = self.root / "runs"
        self.experiment = self.runs / "example_v1"

    @staticmethod
    def epoch(epoch: int, train: str, validation: str) -> dict:
        row = {"epoch": str(epoch)}
        for portion, offset in (("train", 1), ("validation", 2)):
            for index, metric in enumerate(analysis.METRICS):
                row[f"{portion}_{metric}"] = str(Decimal(index + offset) / 10)
        row["train_accuracy"] = train
        row["validation_accuracy"] = validation
        return row

    def combination(self, directory: str, number: int, rows: list[dict],
                    experiment: Path | None = None) -> Path:
        destination = (experiment or self.experiment) / directory
        destination.mkdir(parents=True)
        config = {
            "combination_id": f"combination_{number:02d}",
            "hyperparameters": {
                "learning_rate": 0.0001,
                "lambda_rec": 0.01,
                "lambda_kl": 0.001,
                "lambda_rank": 0.1,
            },
            "architecture": {
                "model_class": "PairwiseCNNVAE",
                "architecture_version": "model_arch_2",
                "latent_dim": 32,
                "dropout": 0.2,
            },
            "batch_size": 16,
            "seed": 42,
            "reporting_seed": 43,
            "epochs": 100,
        }
        (destination / "config.json").write_text(json.dumps(config), encoding="utf-8")
        fields = list(rows[0]) if rows else list(self.epoch(1, "0.5", "0.5"))
        with (destination / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return destination

    @staticmethod
    def record(combination: int, epoch: int, train: str, validation: str) -> dict:
        return {
            "combination_number": combination,
            "combination_id": f"combination_{combination:02d}",
            "epoch": epoch,
            "train_accuracy": train,
            "validation_accuracy": validation,
            "accuracy_gap": abs(Decimal(train) - Decimal(validation)),
        }

    def test_validation_ranking_uses_gap_then_combination_then_epoch(self) -> None:
        records = [
            self.record(2, 9, "0.81", "0.8"),
            self.record(4, 7, "0.80", "0.8"),
            self.record(1, 6, "0.79", "0.8"),
            self.record(1, 3, "0.79", "0.8"),
            self.record(9, 1, "0.91", "0.9"),
        ]
        selected = analysis.rank_records(records, "validation_accuracy")
        self.assertEqual([(r["combination_number"], r["epoch"]) for r in selected],
                         [(9, 1), (4, 7), (1, 3), (1, 6), (2, 9)])

    def test_gap_ranking_uses_validation_then_combination_then_epoch(self) -> None:
        records = [
            self.record(2, 9, "0.81", "0.8"),
            self.record(4, 7, "0.80", "0.8"),
            self.record(1, 6, "0.79", "0.8"),
            self.record(1, 3, "0.79", "0.8"),
            self.record(9, 1, "0.91", "0.9"),
        ]
        selected = analysis.rank_records(records, "accuracy_gap")
        self.assertEqual([(r["combination_number"], r["epoch"]) for r in selected],
                         [(4, 7), (9, 1), (1, 3), (1, 6), (2, 9)])

    def test_repeated_combinations_remain_and_top_n_caps_epochs(self) -> None:
        records = [self.record(1, epoch, "0.9", "0.8") for epoch in range(12, 0, -1)]
        selected = analysis.rank_records(records, "validation_accuracy", top_n=10)
        self.assertEqual([r["epoch"] for r in selected], list(range(1, 11)))
        self.assertEqual(len(analysis.rank_records(records[:2], "accuracy_gap")), 2)

    def test_both_rankings_require_train_and_validation_strictly_above_half(self) -> None:
        records = [
            self.record(1, 1, "0.49", "0.7"),
            self.record(1, 2, "0.5", "0.7"),
            self.record(1, 3, "0.7", "0.49"),
            self.record(1, 4, "0.7", "0.5"),
            self.record(1, 5, "0.5000000000000001", "0.5000000000000001"),
            self.record(1, 6, "0.9", "0.7"),
        ]
        for criterion, expected_epochs in (("validation_accuracy", [6, 5]),
                                           ("accuracy_gap", [5, 6])):
            with self.subTest(criterion=criterion):
                selected = analysis.rank_records(records, criterion)
                self.assertEqual([row["epoch"] for row in selected], expected_epochs)
                self.assertEqual(analysis.rank_records(records[:4], criterion), [])

    def test_load_uses_saved_combination_ids_and_preserves_metric_text(self) -> None:
        first = self.epoch(3, "0.12345678901234567", "0.22345678901234567")
        self.combination("aaa", 11, [first])
        self.combination("zzz", 2, [self.epoch(8, "0.4", "0.3")])
        records = analysis.load_experiment(self.experiment)
        by_number = {r["combination_number"]: r for r in records}
        self.assertEqual(set(by_number), {2, 11})
        record = by_number[11]
        self.assertEqual(record["experiment_version"], "example_v1")
        self.assertEqual(record["combination_id"], "combination_11")
        self.assertEqual(record["epoch"], 3)
        self.assertEqual(record["learning_rate"], 0.0001)
        self.assertEqual(record["lambda_rec"], 0.01)
        self.assertEqual(record["lambda_kl"], 0.001)
        self.assertEqual(record["lambda_rank"], 0.1)
        for key, value in first.items():
            if key != "epoch":
                self.assertEqual(record[key], value)
        self.assertEqual(record["accuracy_gap"], Decimal("0.1"))

    def test_decimal_gap_ties_are_not_broken_by_binary_float_subtraction(self) -> None:
        self.combination("params", 1, [
            self.epoch(1, "0.7", "0.6"),
            self.epoch(2, "0.8", "0.7"),
        ])
        records = analysis.load_experiment(self.experiment)
        self.assertEqual([r["accuracy_gap"] for r in records], [Decimal("0.1")] * 2)
        selected = analysis.rank_records(records, "accuracy_gap")
        self.assertEqual([r["epoch"] for r in selected], [2, 1])

    def test_missing_config_is_reported(self) -> None:
        destination = self.combination("params", 1, [self.epoch(1, "0.5", "0.5")])
        (destination / "config.json").unlink()
        with self.assertRaises(ValueError):
            analysis.load_experiment(self.experiment)

    def test_invalid_metrics_and_duplicate_epochs_are_reported(self) -> None:
        missing = self.epoch(1, "0.5", "0.5")
        missing.pop("validation_kl_divergence")
        nonfinite = self.epoch(1, "0.5", "0.5")
        nonfinite["train_total_loss"] = "NaN"
        duplicate = self.epoch(1, "0.5", "0.5")
        cases = {
            "missing_column": [missing],
            "nonfinite": [nonfinite],
            "duplicate_epoch": [duplicate, duplicate],
            "empty": [],
        }
        for name, rows in cases.items():
            with self.subTest(name=name):
                experiment = self.runs / name
                self.combination("params", 1, rows, experiment)
                with self.assertRaises(ValueError):
                    analysis.load_experiment(experiment)

    def test_duplicate_combination_numbers_are_reported(self) -> None:
        self.combination("one", 1, [self.epoch(1, "0.5", "0.5")])
        self.combination("two", 1, [self.epoch(2, "0.5", "0.5")])
        with self.assertRaises(ValueError):
            analysis.load_experiment(self.experiment)

    def test_plot_has_six_pairs_of_unmodified_metrics(self) -> None:
        self.combination("params", 1, [self.epoch(1, "0.6", "0.7")])
        record = analysis.load_experiment(self.experiment)[0]
        from matplotlib.figure import Figure

        with patch.object(Figure, "savefig", autospec=True) as save:
            analysis.plot_record(record, self.root / "chart.png")
        figure = save.call_args.args[0]
        self.assertEqual(len(figure.axes), 6)
        for axis, metric in zip(figure.axes, analysis.METRICS, strict=True):
            self.assertEqual([bar.get_height() for bar in axis.patches],
                             [float(record[f"train_{metric}"]),
                              float(record[f"validation_{metric}"])])

    def test_softmax_metrics_preserve_components_and_use_categorical_plot_label(self) -> None:
        row = self.epoch(1, "0.6", "0.7")
        for split in ("train", "validation"):
            row.pop(f"{split}_reconstruction_bce")
            row[f"{split}_reconstruction_loss"] = "3.12345678901234567"
            row[f"{split}_tile_cross_entropy"] = "2.92345678901234567"
            row[f"{split}_path_bce"] = "0.2"
        self.combination("softmax", 1, [row])
        record = analysis.load_experiment(self.experiment)[0]
        self.assertNotIn("train_reconstruction_bce", record)
        for split in ("train", "validation"):
            for metric in ("reconstruction_loss", "tile_cross_entropy", "path_bce"):
                self.assertEqual(record[f"{split}_{metric}"], row[f"{split}_{metric}"])
        from matplotlib.figure import Figure

        with patch.object(Figure, "savefig", autospec=True) as save:
            analysis.plot_record(record, self.root / "softmax.png")
        reconstruction_axis = save.call_args.args[0].axes[1]
        self.assertEqual(reconstruction_axis.get_title(), "Reconstruction (tile CE + path BCE)")
        self.assertEqual([bar.get_height() for bar in reconstruction_axis.patches],
                         [float(row[f"{split}_reconstruction_loss"])
                          for split in ("train", "validation")])

    def test_softmax_nonfinite_component_is_reported(self) -> None:
        row = self.epoch(1, "0.6", "0.7")
        for split in ("train", "validation"):
            row[f"{split}_reconstruction_loss"] = row.pop(f"{split}_reconstruction_bce")
            row[f"{split}_path_bce"] = "NaN"
        self.combination("softmax", 1, [row])
        with self.assertRaisesRegex(ValueError, "path_bce"):
            analysis.load_experiment(self.experiment)

    def test_weighted_metrics_retain_multiplier_and_identify_it_in_plots(self) -> None:
        from matplotlib.figure import Figure

        for categorical in (False, True):
            with self.subTest(categorical=categorical):
                experiment = self.runs / str(categorical)
                row = self.epoch(1, "0.6", "0.7")
                if categorical:
                    for split in ("train", "validation"):
                        row[f"{split}_reconstruction_loss"] = row.pop(f"{split}_reconstruction_bce")
                destination = self.combination("weighted", 1, [row], experiment)
                config_path = destination / "config.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                config["hyperparameters"]["background_error_weight"] = 3.0
                config_path.write_text(json.dumps(config), encoding="utf-8")
                record = analysis.load_experiment(experiment)[0]
                self.assertEqual(record["background_error_weight"], 3.0)
                with patch.object(Figure, "savefig", autospec=True) as save:
                    analysis.plot_record(record, self.root / "weighted.png")
                figure = save.call_args.args[0]
                self.assertIn("background_error_weight=3", figure._suptitle.get_text())
                self.assertIn("weighted tile CE" if categorical else "weighted tile BCE",
                              figure.axes[1].get_title())

    def test_invalid_background_error_multiplier_is_reported(self) -> None:
        destination = self.combination("weighted", 1, [self.epoch(1, "0.6", "0.7")])
        config_path = destination / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for multiplier in (0.5, "NaN", "inf", "invalid"):
            with self.subTest(multiplier=multiplier):
                config["hyperparameters"]["background_error_weight"] = multiplier
                config_path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "background_error_weight"):
                    analysis.load_experiment(self.experiment)

    def test_cli_preserves_metrics_for_bce_and_softmax_combinations(self) -> None:
        self.combination("bce", 1, [self.epoch(1, "0.7", "0.6")])
        row = self.epoch(1, "0.8", "0.7")
        for split in ("train", "validation"):
            row[f"{split}_reconstruction_loss"] = row.pop(f"{split}_reconstruction_bce")
            row[f"{split}_tile_cross_entropy"] = "0.12345678901234567"
            row[f"{split}_path_bce"] = "0.2"
        destination = self.combination("softmax", 2, [row])
        config_path = destination / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["hyperparameters"]["background_error_weight"] = 3.0
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output = self.root / "analysis"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(analysis.main(["--runs-dir", str(self.runs),
                                            "--output-dir", str(output)]), 0)
        with (output / self.experiment.name / "top_validation_accuracy.csv").open(
                newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        softmax, bce = records
        self.assertEqual(softmax["background_error_weight"], "3.0")
        self.assertEqual(bce["background_error_weight"], "")
        self.assertEqual(softmax["train_tile_cross_entropy"], "0.12345678901234567")
        self.assertEqual(softmax["validation_path_bce"], "0.2")
        self.assertEqual(softmax["train_reconstruction_bce"], "")
        self.assertEqual(bce["train_reconstruction_bce"], "0.2")
        self.assertEqual(bce["train_reconstruction_loss"], "")

    def test_cli_writes_both_rankings_and_every_selected_png(self) -> None:
        self.combination("params", 7, [
            self.epoch(1, "0.6", "0.6"),
            self.epoch(2, "0.8", "0.7"),
            self.epoch(3, "0.9", "0.8"),
        ])
        self.combination("other", 2, [self.epoch(1, "0.7", "0.7")],
                         self.runs / "example_v2")
        output = self.root / "analysis"
        with redirect_stdout(io.StringIO()):
            result = analysis.main(["--runs-dir", str(self.runs), "--output-dir", str(output),
                                    "--top-n", "2"])
        self.assertEqual(result, 0)
        from PIL import Image

        for version, expected_count in (("example_v1", 2), ("example_v2", 1)):
            for filename in ("top_validation_accuracy.csv", "smallest_accuracy_gap.csv"):
                with (output / version / filename).open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), expected_count)
                self.assertEqual([row["rank"] for row in rows],
                                 [str(rank) for rank in range(1, expected_count + 1)])
                for row in rows:
                    self.assertEqual(row["experiment_version"], version)
                    for metric in analysis.METRICS:
                        for portion in ("train", "validation"):
                            self.assertNotEqual(row[f"{portion}_{metric}"], "")
                    with Image.open(output / version / row["plot_file"]) as chart:
                        self.assertEqual(chart.format, "PNG")
                        self.assertGreater(chart.width, 500)
                        self.assertGreater(chart.height, 500)
                        chart.verify()

    def test_cli_handles_fresh_experiment_with_no_eligible_epochs(self) -> None:
        self.combination("params", 1, [self.epoch(1, "0.5", "0.7"),
                                       self.epoch(2, "0.49", "0.7"),
                                       self.epoch(3, "0.7", "0.5"),
                                       self.epoch(4, "0.7", "0.49")])
        output = self.root / "analysis"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(analysis.main(["--runs-dir", str(self.runs),
                                            "--output-dir", str(output)]), 0)
        report_directory = output / self.experiment.name
        for filename in ("top_validation_accuracy.csv", "smallest_accuracy_gap.csv"):
            with (report_directory / filename).open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertIn("validation_accuracy", reader.fieldnames)
                self.assertEqual(list(reader), [])
        self.assertEqual(list(report_directory.rglob("*.png")), [])
        self.assertIn("0 eligible", (report_directory / "README.md").read_text(encoding="utf-8"))

    def test_cli_refresh_with_no_eligible_epochs_removes_only_old_generated_plots(self) -> None:
        destination = self.combination("params", 1, [self.epoch(1, "0.7", "0.6")])
        output = self.root / "analysis"
        arguments = ["--runs-dir", str(self.runs), "--output-dir", str(output)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(analysis.main(arguments), 0)
        report_directory = output / self.experiment.name
        generated_pngs = set(report_directory.glob("plots/*.png"))
        self.assertEqual(len(generated_pngs), 1)
        unrelated = report_directory / "plots" / "my_notes.png"
        unrelated.write_bytes(b"Unrelated file to preserve")

        rows = [self.epoch(1, "0.5", "0.7"), self.epoch(2, "0.49", "0.7"),
                self.epoch(3, "0.7", "0.5"), self.epoch(4, "0.7", "0.49")]
        with (destination / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(analysis.main(arguments), 0)

        for filename in ("top_validation_accuracy.csv", "smallest_accuracy_gap.csv"):
            with (report_directory / filename).open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertTrue({"rank", "experiment_version", "combination_id", "epoch",
                                 "accuracy_gap", *analysis.METRIC_COLUMNS} <= set(reader.fieldnames))
                self.assertEqual(list(reader), [])
        self.assertTrue(all(not png.exists() for png in generated_pngs))
        self.assertEqual(set(report_directory.glob("plots/*.png")), {unrelated})
        self.assertEqual(unrelated.read_bytes(), b"Unrelated file to preserve")
        self.assertIn("0 eligible", (report_directory / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
