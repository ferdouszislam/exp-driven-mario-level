# Hyperparameter comparison

Each configuration trains for 100 epochs.
The fixed group split contains 232 fitting comparisons and 58 validation comparisons.
Training metrics refer to the fitting portion. Both portions come from train_set.
Choose a configuration and epoch from its loss curves and per-epoch metrics.

| Configuration | Learning rate | Rec weight | KL weight | Rank weight | Epochs | Final train accuracy | Final validation accuracy | Checkpoints | Loss graph |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| combination_01 | 0.001 | 1.0 | 0.001 | 1.0 | 100 | 1.000000 | 0.422414 | 80 | [Curves](combination_01/loss_curves.png) |
| combination_02 | 0.001 | 2.0 | 0.001 | 1.0 | 100 | 1.000000 | 0.336207 | 80 | [Curves](combination_02/loss_curves.png) |
| combination_03 | 0.001 | 1.0 | 0.001 | 2.0 | 100 | 1.000000 | 0.543103 | 80 | [Curves](combination_03/loss_curves.png) |
| combination_04 | 0.001 | 1.0 | 0.01 | 1.0 | 100 | 0.997845 | 0.465517 | 80 | [Curves](combination_04/loss_curves.png) |
| combination_05 | 0.001 | 2.0 | 0.01 | 1.0 | 100 | 1.000000 | 0.508621 | 80 | [Curves](combination_05/loss_curves.png) |
| combination_06 | 0.001 | 1.0 | 0.01 | 2.0 | 100 | 1.000000 | 0.482759 | 80 | [Curves](combination_06/loss_curves.png) |
| combination_07 | 0.0003 | 1.0 | 0.001 | 1.0 | 100 | 1.000000 | 0.568966 | 80 | [Curves](combination_07/loss_curves.png) |
| combination_08 | 0.0003 | 2.0 | 0.001 | 1.0 | 100 | 0.997845 | 0.543103 | 80 | [Curves](combination_08/loss_curves.png) |
| combination_09 | 0.0003 | 1.0 | 0.001 | 2.0 | 100 | 1.000000 | 0.439655 | 80 | [Curves](combination_09/loss_curves.png) |
| combination_10 | 0.0003 | 1.0 | 0.01 | 1.0 | 100 | 0.997845 | 0.500000 | 80 | [Curves](combination_10/loss_curves.png) |
| combination_11 | 0.0003 | 2.0 | 0.01 | 1.0 | 100 | 0.997845 | 0.517241 | 80 | [Curves](combination_11/loss_curves.png) |
| combination_12 | 0.0003 | 1.0 | 0.01 | 2.0 | 100 | 0.997845 | 0.534483 | 80 | [Curves](combination_12/loss_curves.png) |

Each graph contains total loss, reconstruction BCE, KL divergence, and ranking BCE panels.
The panels show fitting and validation curves against num_epochs.
Each configuration's train_history.json and epoch_metrics.csv record every epoch's losses,
overall accuracy and binary F1.

Checkpoints cover epochs 21–100, inclusive.
Each checkpoint filename identifies its saved epoch, such as epoch_021.pt for epoch 21.
