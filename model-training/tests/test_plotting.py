"""Check reconstruction labels and curves for both output/loss conventions."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plotting import plot_loss_curves


class PlottingTests(unittest.TestCase):
    def test_reconstruction_panel_uses_saved_loss_convention(self):
        from matplotlib.figure import Figure

        for metric, title in (
            ("reconstruction_bce", "Reconstruction BCE"),
            ("reconstruction_loss", "Reconstruction (tile CE + path BCE)"),
        ):
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as directory:
                measurements = {
                    "total_loss": 4.0, metric: 3.5, "kl_divergence": 0.1,
                    "ranking_bce": 0.4, "accuracy": 0.75,
                    "normalized_hamming_distance": 0.25,
                }
                history = [{"epoch": 1, "train": measurements,
                            "validation": {**measurements, metric: 3.7}}]
                parameters = {"learning_rate": 1e-4, "lambda_rec": 1.0,
                              "lambda_kl": 0.001, "lambda_rank": 1.0}
                with patch.object(Figure, "savefig", autospec=True) as save:
                    plot_loss_curves(history, parameters, Path(directory) / "curves.png", "combination_01")
                figure = save.call_args.args[0]
                self.assertEqual(len(figure.axes), 6)
                reconstruction = figure.axes[1]
                self.assertEqual(reconstruction.get_title(), title)
                self.assertEqual([list(line.get_ydata()) for line in reconstruction.lines],
                                 [[3.5], [3.7]])


if __name__ == "__main__":
    unittest.main()
