# Preprocessing Mario-Fun-Study for pairwise fun comparisons

[`preprocess_dataset.py`](preprocess_dataset.py) converts the supplied cleaned human annotations into comparison folders. It first assigns an 80/20 training/test split, then holds out 20% of that training portion in `validation_set/`, using the same whole-group method as [`train.py`](../../model-training/initial_model_architecture/train.py). Each folder holds two cropped tile grids, their participant-specific path masks, their multi-hot NumPy arrays, and a two-line preference file. The first 16 source columns are removed from both tiles and paths, leaving 176 columns. The full dataset audit and paper discrepancies are documented in [`dataset-explained.md`](dataset-explained.md).

The source is `Mario-Fun-Study/exp_data/survey_data/formal/` and the default output is `preprocessed_dataset/`. Run the commands below from this project directory.

## Setup and reproduction

Use Python 3.11 or newer; this export was generated and checked with Python 3.14.7. Reuse the repository-root `.venv` with the preprocessing [requirements.txt](requirements.txt). Create the environment once from the repository root, then install these dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r datasets/fun-study-dataset-analysis/requirements.txt
```

Subsequent short script commands run from `datasets/fun-study-dataset-analysis/` and address the root interpreter as `../../.venv/bin/python`. The dependency file pins `numpy==2.3.4` and `Pillow==12.0.0` for preprocessing and image rendering. If `.venv` already exists, reuse it and run the installation command above; no separate environment is needed for either script.

Generate the dataset with the default split seed, `42`:

```bash
../../.venv/bin/python preprocess_dataset.py
```

Verify the generated files against the source annotations, maps, and human paths:

```bash
../../.venv/bin/python preprocess_dataset.py --verify-only
```

Choose another output directory when needed:

```bash
../../.venv/bin/python preprocess_dataset.py --output my_preprocessed_dataset
../../.venv/bin/python preprocess_dataset.py --output my_preprocessed_dataset --verify-only
```

Use the same seed when generating and verifying another split:

```bash
../../.venv/bin/python preprocess_dataset.py --split-seed 7 --output dataset_seed7
../../.venv/bin/python preprocess_dataset.py --split-seed 7 --output dataset_seed7 --verify-only
```

`--source` can select another copy of the formal study directory. The source and output must be separate directories, with neither nested inside the other. With no flags, their locations are resolved relative to the script, so the defaults also work when invoked from another directory.

An existing output is validated against the expected source-derived contents and split seed. An identical rerun leaves it in place; a mismatch fails instead of overwriting it. Use a new output directory when intentionally changing inputs or preprocessing policy, or explicitly rebuild the existing export as described below.

To recalculate an existing 192-column export with the current 16-column crop, run from the repository root:

```bash
.venv/bin/python datasets/fun-study-dataset-analysis/preprocess_dataset.py --rebuild
.venv/bin/python datasets/fun-study-dataset-analysis/preprocess_dataset.py --verify-only
.venv/bin/python datasets/fun-study-dataset-analysis/encoding_to_level_image.py dataset
```

`--rebuild` constructs and verifies the new export in a temporary sibling directory before replacing the selected output. It regenerates all three splits from source with the requested seeds and fractions. Old PNG previews are discarded and can be recreated with the renderer. The default seeds preserve comparison memberships and rankings; the crop changes the tile/path files, tensors, and their metadata.

To add validation to an existing export that already uses the current crop, or migrate its earlier flat layout, run once:

```bash
../../.venv/bin/python preprocess_dataset.py --migrate-layout
../../.venv/bin/python preprocess_dataset.py --verify-only
```

Migration first checks all existing comparison files against content reconstructed from the source. For a train/test export it moves only the selected training folders into `validation_set/` and updates the global metadata. The seven files inside each comparison folder keep their contents, and the test split keeps its membership and contents. Use `--output` and `--split-seed` with migration when selecting a different output path or seed. Normal generation does not silently migrate an existing export.

From the repository root, run the dataset preparation script with:

```bash
.venv/bin/python datasets/fun-study-dataset-analysis/preprocess_dataset.py --migrate-layout
```

Use `--validation-fraction 0.2` for the default 80/20 split of the original training portion. For a different ratio, supply the same fraction during generation and verification, for example:

```bash
../../.venv/bin/python preprocess_dataset.py --validation-fraction 0.1 --output dataset_val10
../../.venv/bin/python preprocess_dataset.py --validation-fraction 0.1 --output dataset_val10 --verify-only
```

`--validation-fraction 0` retains the earlier train/test-only layout.

## Which annotations become instances?

The script reads the **supplied cleaned** `annotation.csv` and `annotation2.csv`. It applies no additional completion threshold and does not require the player to have won.

| Source | Conversion | Result |
|---|---|---:|
| Experiment I, `annotation.csv` | Keep `A` or `B` preferences; put the preferred level first | 90 comparisons |
| Experiment I, `annotation.csv` | Exclude 17 `E` (equally fun) and 17 `N` (neither fun) rows because they do not give a strict preferred order | 34 exclusions |
| Experiment II, `annotation2.csv` | Expand each of 91 ordered triples into all three preferred pairs | 273 comparisons |
| Total | One folder per strict comparison | **363 comparison folders** |

For a ranking `first > second > third`, the three derived pairs are `first > second`, `first > third`, and `second > third`, in that order. Resolve `R1`, `R2`, and `R3` through that row's concrete `R`, `K`, and `C` level IDs. They are ordering codes, not file names or numerical fun scores.

The three-level ranking data sometimes referred to as “experiment 3” is **Experiment II in the paper**, stored in `annotation2.csv`. The output uses `exp2` to match the published experiment names.

The 363 folders cover **450 unique participant–level playthroughs, 290 distinct maps, and 84 participants**. There are 726 item appearances because each comparison contains two items. The original cleaned release has 518 playthroughs over 313 maps and 90 participants; omitting the 34 nondirectional Experiment I judgments removes 68 playthroughs from this strict-comparison export.

Exclusion applies to an annotation row, not a global map blacklist. The excluded rows contain 62 map IDs, but 39 of those maps remain through other retained judgments. Only 23 distinct geometries disappear from the export, accounting for the change from 313 to 290 maps.

The repeated Experiment II ranking is retained. It is an additional source row that reuses the same three human paths, so the 273 derived pairs are not 273 independent submissions. Source grouping and reused playthrough identities are retained in the global metadata.

## Directory structure and the seven core files

The output has **232 comparison folders in `train_set/`, 58 in `validation_set/`, and 73 in `test_set/`** for the supplied data and default settings. Each comparison contains seven core data files and can include two derived PNG previews:

```text
preprocessed_dataset/
  channels.json
  manifest.json
  summary.json
  excluded_comparisons.json
  source_hashes.json
  train_set/
    <instance_id>/
      <preferred_level>.tiles.txt
      <preferred_level>.path.txt
      <preferred_level>.npy
      <preferred_level>.png  (optional preview)
      <other_level>.tiles.txt
      <other_level>.path.txt
      <other_level>.npy
      <other_level>.png      (optional preview)
      ranking.txt
    ...
  validation_set/
    <instance_id>/
      ...the same seven core files and optional previews...
    ...
  test_set/
    <instance_id>/
      ...the same seven core files and optional previews...
    ...
```

Instance IDs retain names such as `exp1_0001__lvl89_over_lvl134` and `exp2_0001_pair1__Runner-16_over_Killer-29`. Their assigned split is recorded in the manifest; do not infer it from the instance name.

Each instance contains **seven core data files**, plus two PNGs after preview generation. The five global metadata files remain at the root. In each folder:

| File | Contents |
|---|---|
| `<level>.tiles.txt` | 16 rows of 176 literal tile characters; source columns `16:192` |
| `<level>.path.txt` | 16 rows of 176 characters: `P` for a visited path-marker cell and `x` otherwise |
| `<level>.npy` | A `uint8` multi-hot array with shape `(22, 16, 176)` |
| `ranking.txt` | Two concrete level IDs, one per line, preferred first |
| `<level>.png` | Optional 2816×256 preview of the corresponding encoding, including black X path markers at the default scale |

Generate the previews using the [existing renderer](ENCODING_TO_LEVEL_IMAGE.md). From the repository root, run:

```bash
.venv/bin/python datasets/fun-study-dataset-analysis/encoding_to_level_image.py dataset
```

This adds **726 PNGs**: 464 in `train_set/`, 116 in `validation_set/`, and 146 in `test_set/`. The original data files and split assignments are preserved.

For `exp1_0001__lvl89_over_lvl134`, `ranking.txt` contains:

```text
lvl89
lvl134
```

The path belongs to the participant who submitted that comparison. If another participant played the same map, its tile text is identical but its path and combined array can differ. A level ID alone therefore does not identify a training playthrough.

Folder numbers are the one-based data-row positions in the cleaned source CSVs, before exclusions. Experiment I numbering therefore has gaps. Each Experiment II row keeps its number across its three `pair1`–`pair3` folders.

The 726 NumPy files contain approximately 45 MB of array data in total. Repeated appearances are deliberately stored within their comparison folders for self-contained loading.

## Tile and path encoding

The source grids have **192 columns**. The export drops the first **16 columns** from every map, retaining source columns `16:192` as **176 exported columns**. The same crop applies to the path mask. Exported column 0 corresponds to source column 16, or source x = 256 pixels. Rows increase downward and columns increase rightward from the cropped upper-left origin; source maps and trace coordinates remain unchanged.

Each human trace contains ordered pixel positions `[x, y]`. Each sampled position maps to:

```text
column = floor(x / 16) - 16
row    = floor((y - 8) / 16)
```

This follows the repository's drawn trace-marker convention. A sample marks its cell only if `0 <= row < 16` and `0 <= column < 176`. Samples in the removed prefix and out-of-grid samples are counted and skipped; they are not clipped onto border cells. The converter does not interpolate between samples. Repeated visits set the same bit again.

In this export's 450 unique playthroughs, **1,903 of 764,548 recorded samples** fall outside the original 192-column source grid, across **203 playthroughs**. `out_of_grid_samples` retains that source-frame meaning. `cropped_prefix_samples` separately counts **83,908** otherwise-valid samples removed with columns 0–15; `outside_export_samples` is **85,811**, their sum with the original out-of-grid count. The cropped export retains **678,737 valid samples** across **153,653 visited cells**. `valid_samples` and `visited_cells` refer to that cropped frame. These counts are reported once per unique playthrough in `summary.json`, rather than being inflated by repeated item appearances.

The path text is a binary occupancy mask: it discards sample order, repeated visits, duration, actions, and any path outside the stored grid. Original ordered positions remain available in the source `formal/res/` files and are linked through the manifest. A `P` identifies the recorded marker cell, not Mario's full collision footprint.

The array is **channels first**: `array[channel, row, column]`. Channels 0–20 preserve the 21 literal tile characters found across the main study maps; channel 21 is the path bit:

| Channel | Symbol | Meaning |
|---:|:---:|---|
| 0 | `X` | Solid tile / ground |
| 1 | `S` | Breakable block |
| 2 | `-` | Empty tile |
| 3 | `%` | Mushroom platform |
| 4 | `t` | Normal pipe |
| 5 | `T` | Flower pipe |
| 6 | `o` | Coin |
| 7 | `Q` | Coin question block |
| 8 | `@` | Mushroom question block |
| 9 | `U` | Mushroom breakable block |
| 10 | `L` | 1UP block |
| 11 | `1` | Invisible 1UP block |
| 12 | `2` | Invisible coin block |
| 13 | `g` | Goomba |
| 14 | `k` | Green Koopa |
| 15 | `r` | Red Koopa |
| 16 | `K` | Flying green Koopa |
| 17 | `R` | Flying red Koopa |
| 18 | `y` | Spiky |
| 19 | `#` | Literal solid alias, kept distinct from `X` |
| 20 | `\|` | Literal mushroom stalk, kept distinct from empty |
| 21 | `PATH` | Visited path-marker cell; corresponds to `P` in path text |

This channel order is the export's codebook. It differs from the repository's canonical model IDs, which merge aliases and include absent Bullet Bill classes `b` and `B`. Read the saved `channels.json` when loading tensors; do not substitute another channel order from the explanatory report or repository encoder.

Every cell has exactly one active tile bit, including empty cells. A path visit adds one active bit without replacing the tile:

```text
No path visit: sum(array[:, row, column]) = 1
Path visit:    sum(array[:, row, column]) = 2
```

Thus a path through empty space activates both the empty-tile channel and the path channel. All values are zero or one, stored as NumPy `uint8`.

## Loading one comparison

```python
import json
from pathlib import Path
import numpy as np

root = Path("preprocessed_dataset")
manifest = json.loads((root / "manifest.json").read_text())
entry = next(item for item in manifest["instances"] if item["split"] == "train_set")
instance = root / entry["relative_directory"]
preferred, other = (instance / "ranking.txt").read_text().splitlines()

preferred_array = np.load(instance / f"{preferred}.npy", allow_pickle=False)
other_array = np.load(instance / f"{other}.npy", allow_pickle=False)

assert preferred_array.shape == other_array.shape == (22, 16, 176)
assert preferred_array.dtype == other_array.dtype == np.uint8
assert np.all(preferred_array[:21].sum(axis=0) == 1)
assert set(np.unique(preferred_array)).issubset({0, 1})

path_mask = preferred_array[21]   # (16, 176), one at recorded path cells
tile_channels = preferred_array[:21]
```

`ranking.txt` gives a relative preference for this particular comparison. The first item has no absolute numerical fun value, and it is not automatically better than a first-ranked item in another folder.

## Provenance, verification, and training splits

The five root JSON files describe the export:

| File | Purpose |
|---|---|
| `channels.json` | Exact channel order, tile meanings, exported/source shapes, crop, dtype, and path text symbols |
| `manifest.json` | `instances` list with split assignments, relative paths, comparison-to-source mapping, participant identities, original annotation groups, and concrete level/playthrough references |
| `summary.json` | Counts, split details, path statistics, preprocessing policies, script hash, and NumPy version |
| `excluded_comparisons.json` | Original Experiment I rows excluded because their label was `E` or `N` |
| `source_hashes.json` | Source-file hashes used to check provenance and reruns |

Verification checks the expected core folder/file inventory and reconstructs the expected content from source. It compares tile and path text, preference order, and loaded NumPy arrays, including their shape, dtype, binary values, and tile/path consistency. It also checks global metadata and source hashes. Optional PNG counterparts of expected `.npy` files are allowed alongside the core files; unrelated extra files are rejected. PNGs are excluded from the core dataset digest. Their rendered pixels can be validated separately against a fresh render of each encoding; `--verify-only` checks the core data and permits these preview files. It requires an existing export and performs no generation.

The script reports `created_and_verified` after a new export, `rebuilt_and_verified` after an explicit rebuild, `migrated_and_verified` after moving a checked flat export, or `verified_existing` after a matching rerun or verification. It also reports the train/validation/test counts, seed, core file count, and core dataset SHA-256. This export contains **2,546 core files**: `363 × 7` instance files plus five root metadata files. With all 726 PNG previews, the directory contains **3,272 files**. The recorded script hash and NumPy version are part of the expected metadata, so changing either can make a previous export fail exact verification even when the intended preprocessing policy is unchanged.

For this export, a fresh generation into another directory produced byte-identical contents across all 2,546 files. Rerunning against that existing output validated it without changing file modification times. A deliberately altered ranking file in a temporary copy was rejected by verification and left untouched. Independently reconstructed source comparisons, masks, and tensors also matched every generated instance.

The split layout keeps schema version **2** for `manifest.json`; the cropped `summary.json` uses schema version **4**, and `channels.json` uses schema version **2** with source dimensions and crop metadata. Each manifest entry has `split` (`train_set`, `validation_set`, or `test_set`), `split_group_id` (its connected group), and `relative_directory` (for example, `validation_set/<instance_id>`). Its `ranking_file` and per-playthrough `files` paths are also relative to the output root. The entry selected in the loading example can be inspected directly:

```python
assert entry["instance_id"] == instance.name
print(entry["split"], entry["relative_directory"])
print(entry["split_group_id"])
print(entry["participant_id"])
print(entry["source_annotation_id"])
print(entry["annotation_reuse_group"])
for play in entry["playthroughs"]:
    print(play["playthrough_key"], play["source_result"], play["trace_stats"])
```

## How the grouped splits are chosen

For `N` comparison folders, the target training size is `floor(4*N/5)` and the target test size is the remainder. Here `N=363`, so the targets are **290 train / 73 test**, or approximately **79.89% / 20.11%**.

The splitter first connects comparisons that share any of these identities, including connections reached through other comparisons:

- `source_annotation_id`: all three pairs derived from one three-level ranking stay together.
- `annotation_reuse_group`: repeated submissions of the same ordering stay together.
- `playthrough_key`: every reuse of the same participant's path through a particular level stays together.

It assigns whole connected groups to a split. A seed-dependent SHA-256 ordering and subset-sum selection choose a test set whose size is closest to the target without breaking any group. The supplied data permits the exact target of 73 test comparisons. For changed input data, group sizes can prevent an exact 80/20 count; `summary.json` records the achieved sizes.

The default seed is **42**. Identical source files, script, dependency, and seed reproduce the same split. Changing `--split-seed` can change membership while retaining the grouping rules; use a new output directory for that change.

Validation is drawn only from the original 290 training comparisons. As in `train.py`, the exporter sorts `split_group_id` values, shuffles them with `random.Random(seed)`, and takes whole groups until reaching `max(1, round(original_training_count * validation_fraction))`. It always leaves at least one complete group for training. With fewer than two groups, all comparisons remain in training. Group sizes can make the achieved ratio differ from the target. This second split is recorded in `summary.json` → `split` → `validation`; the original train/test audit remains under `original_train_test_split`.

The default seed produces these counts, saved under `summary.json` → `split`:

| Measure | Train | Validation | Test |
|---|---:|---:|---:|
| Comparison folders, `instances` | 232 | 58 | 73 |
| Experiment I comparisons, `by_experiment` → `1` | 67 | 10 | 13 |
| Experiment II comparisons, `by_experiment` → `2` | 165 | 48 | 60 |
| Connected groups | 122 | 26 | 32 |
| Unique playthroughs, `unique_playthroughs` | 299 | 68 | 83 |

The 80/20 target applies to comparison folders overall, rather than separately to each experiment, participant, or map.

The delivered splits are **playthrough-disjoint**: no `split_group_id`, source annotation, repeated-annotation group, or `playthrough_key` crosses any pair of sets. Participant IDs and map IDs may still recur through different playthroughs; `shared_between_sets` reports these overlaps for each pair of splits. It measures transfer to held-out recorded playthroughs; it is not a participant-held-out or level-held-out evaluation. Training consumes the saved validation membership directly; its `--validation-fraction` applies only to older exports without a saved validation set.

These folders encode human preferences over experienced level/path combinations. If a generator will produce only level geometry, decide how its path will be obtained at inference time; a simulated path and the human path in this dataset are different sources. The export supplies training data and provenance, without converting relative rankings into absolute fun ratings.
