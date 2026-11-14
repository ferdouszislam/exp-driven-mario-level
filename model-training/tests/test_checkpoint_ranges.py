"""Check checkpoint-range CLI behavior for every local training pipeline."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_weighted_pipelines import TRAINING_ROOT, pipeline

sys.path.insert(0, str(TRAINING_ROOT))

from checkpoint_ranges import checkpoint_summary


BASE_ARCHITECTURES = ("initial_model_architecture", "model_arch_1", "model_arch_2")
PIPELINES = tuple(base + suffix for suffix in ("", "_softmax", "_weighted", "_softmax_weighted")
                  for base in BASE_ARCHITECTURES)


class CheckpointRangeTests(unittest.TestCase):
    def test_report_describes_actual_range_union_within_epoch_budget(self):
        self.assertEqual(checkpoint_summary([(4, 9), (1, 2), (4, 4), (8, 12)], 6),
                         "Checkpoints cover epochs 1–2, 4–6, inclusive.")
        self.assertEqual(checkpoint_summary([(8, 12), (7, 9)], 6),
                         "Checkpoint saving begins at epoch 7, beyond this run's 6-epoch budget.")
        self.assertEqual(checkpoint_summary([(3, 3), (1, 2), (2, 3)], 6),
                         "Checkpoints cover epochs 1–3, inclusive.")

    def test_all_pipelines_accept_single_repeated_and_list_ranges(self):
        cases = (
            ([], None, None, []),
            (["--checkpoint-start-epoch", "2", "--checkpoint-end-epoch", "3"],
             [2], [3], [(2, 3)]),
            (["--checkpoint-start-epoch", "2", "--checkpoint-end-epoch", "3",
              "--checkpoint-start-epoch", "8", "--checkpoint-end-epoch", "10"],
             [2, 8], [3, 10], [(2, 3), (8, 10)]),
            (["--checkpoint-start-epoch", "8", "2", "8", "--checkpoint-end-epoch", "10", "3", "8"],
             [8, 2, 8], [10, 3, 8], [(8, 10), (2, 3), (8, 8)]),
            (["--checkpoint-start-epoch", "1", "3", "--checkpoint-end-epoch", "2",
              "--checkpoint-start-epoch", "9", "--checkpoint-end-epoch", "4", "12"],
             [1, 3, 9], [2, 4, 12], [(1, 2), (3, 4), (9, 12)]),
        )
        for name in PIPELINES:
            with self.subTest(pipeline=name), pipeline(name) as (_, training):
                for flags, starts, ends, ranges in cases:
                    with self.subTest(flags=flags):
                        args = training.parse_args(["--epochs", "5", *flags])
                        self.assertEqual(args.checkpoint_start_epoch, starts)
                        self.assertEqual(args.checkpoint_end_epoch, ends)
                        self.assertEqual(args.checkpoint_ranges, ranges)

    def test_constants_accept_integer_and_list_ranges_and_cli_replaces_defaults(self):
        for name in PIPELINES:
            with self.subTest(pipeline=name), pipeline(name) as (_, training):
                for starts, ends, expected in ((2, 3, [(2, 3)]),
                                               ([1, 8], [3, 10], [(1, 3), (8, 10)])):
                    with self.subTest(starts=starts, ends=ends):
                        with patch.object(training, "CHECKPOINT_START_EPOCH", starts), \
                                patch.object(training, "CHECKPOINT_END_EPOCH", ends):
                            args = training.parse_args([])
                            self.assertEqual(args.checkpoint_ranges, expected)
                            overridden = training.parse_args([
                                "--checkpoint-start-epoch", "4", "--checkpoint-end-epoch", "6"])
                            self.assertEqual(overridden.checkpoint_start_epoch, [4])
                            self.assertEqual(overridden.checkpoint_end_epoch, [6])
                            self.assertEqual(overridden.checkpoint_ranges, [(4, 6)])
                            # Parsing flags must leave edited module constants intact.
                            self.assertEqual(training.CHECKPOINT_START_EPOCH, starts)
                            self.assertEqual(training.CHECKPOINT_END_EPOCH, ends)
                            self.assertEqual(training.parse_args([]).checkpoint_ranges, expected)

    def test_invalid_ranges_fail_before_data_or_outputs(self):
        invalid = (
            ["--checkpoint-start-epoch", "2"],
            ["--checkpoint-end-epoch", "3"],
            ["--checkpoint-start-epoch", "1", "4", "--checkpoint-end-epoch", "3"],
            ["--checkpoint-start-epoch", "1", "--checkpoint-end-epoch", "3", "5"],
            ["--checkpoint-start-epoch", "1", "0", "--checkpoint-end-epoch", "3", "5"],
            ["--checkpoint-start-epoch", "1", "4", "--checkpoint-end-epoch", "3", "0"],
            ["--checkpoint-start-epoch", "1", "-2", "--checkpoint-end-epoch", "3", "5"],
            ["--checkpoint-start-epoch", "1", "4", "--checkpoint-end-epoch", "3", "-5"],
            ["--checkpoint-start-epoch", "1", "7", "--checkpoint-end-epoch", "3", "5"],
            ["--checkpoint-start-epoch", "--checkpoint-end-epoch", "3"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "should-not-exist"
            for name in PIPELINES:
                with self.subTest(pipeline=name), pipeline(name) as (_, training):
                    with patch.object(training, "load_training_data") as load_data, \
                            patch.object(training, "load_hyperparameter_combinations") as load_rows:
                        for flags in invalid:
                            with self.subTest(flags=flags), redirect_stderr(io.StringIO()):
                                with self.assertRaises(SystemExit) as rejected:
                                    training.main([*flags, "--output-dir", str(output)])
                                self.assertEqual(rejected.exception.code, 2)
                        load_data.assert_not_called()
                        load_rows.assert_not_called()
                        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
