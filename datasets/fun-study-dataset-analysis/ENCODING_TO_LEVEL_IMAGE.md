# Render level images from tile/path encodings

[`encoding_to_level_image.py`](encoding_to_level_image.py) decodes a channel-first NumPy encoding and renders its Mario tiles with the supplied sprite assets. A **black X marks every visited path cell**. The renderer uses the encoding itself, the channel codebook, and sprites; it does not read companion `.tiles.txt` / `.path.txt` files or copy an existing level image.

The examples below show **three samples from `train_set` and three from `test_set`**, six images in total. Each sample is a distinct participant–level playthrough. Full source references are also recorded in [`encoding_level_images/samples.json`](encoding_level_images/samples.json).

## Install and reproduce

Run from this project directory, using the shared root `.venv` described in [`PREPROCESSING.md`](PREPROCESSING.md):

```bash
.venv/bin/python -m pip install -r requirements.txt
```

The single project [requirements.txt](requirements.txt) pins `numpy==2.3.4` and `Pillow==12.0.0`. Preprocessing and rendering use the same environment; reuse the existing `.venv`.

Render both levels in every comparison folder across `train_set/`, `validation_set/`, and `test_set/`. Run this command from the repository root using its existing `.venv`:

```bash
.venv/bin/python datasets/fun-study-dataset-analysis/encoding_to_level_image.py dataset
```

The default dataset is resolved relative to the renderer. From its directory, an explicit dataset path can be supplied with `encoding_to_level_image.py dataset --dataset preprocessed_dataset`; the command also accepts `--assets DIR` and `--scale N`.

Each `<level>.npy` receives a sibling `<level>.png`, rendered with black X path markers. At the default `--scale 1`, each image is 3072×256 pixels. The supplied export has **726 PNG previews**: 464 in `train_set/`, 116 in `validation_set/`, and 146 in `test_set/`. Every comparison folder has two previews, including repeated appearances of a playthrough in different comparisons. The seven core data files, manifest, and core dataset digest retain their roles; previews are optional derived files.

Recreate the six examples:

```bash
.venv/bin/python encoding_to_level_image.py examples \
  --dataset preprocessed_dataset \
  --output-dir encoding_level_images \
  --count 3 --seed 42 --scale 1
```

`--count 3` selects **three from each split**, not three overall. Selection uses one `random.Random(42)` stream, sampling training first and test second. Within each split, candidates are unique `playthrough_key` values in sorted order. When the same playthrough occurs in several comparison folders, the lexicographically first instance supplies its encoding. Repeated appearances therefore do not give that playthrough extra sampling weight.

The `examples` command keeps its sample collection outside `preprocessed_dataset/`. Use the `dataset` command above for PNG previews beside every encoded level.

Render one saved encoding:

```bash
.venv/bin/python encoding_to_level_image.py render \
  preprocessed_dataset/train_set/exp1_0045__lvl195_over_lvl104/lvl195.npy \
  lvl195-with-path.png
```

Use `--no-path` to omit the X overlay or `--scale 2` for a larger image:

```bash
.venv/bin/python encoding_to_level_image.py render \
  preprocessed_dataset/train_set/exp1_0045__lvl195_over_lvl104/lvl195.npy \
  lvl195-tiles-only.png --no-path --scale 2
```

The `render` command also accepts `--channels FILE` and `--assets DIR`. Defaults are `preprocessed_dataset/channels.json` and `Mario-Fun-Study/smb/assets/`, resolved relative to the renderer's project directory. Use the codebook that matches your tensor's channel order.

## Inputs and rendering conventions

Accepted arrays have shape **`(21, H, W)`** for tile-only one-hot data or **`(22, H, W)`** for the tile-plus-path export. `H` and `W` can be any positive sizes. With the default codebook, channels 0–20 are:

```text
X S - % t T o Q @ U L 1 2 g k r K R y # |
```

Channel 21 is `PATH`. All values must be binary, and each cell must have exactly one active tile channel. A multi-hot input can additionally activate its path bit. A tile-only input produces an all-false path mask. The decoder rejects nonbinary values, missing tile identities, and multiple active tile identities instead of silently choosing one.

Decoding considers **only the tile channels** when recovering the character grid. The path channel is separate, so a visited tile keeps its original identity. This renderer expects the export's literal character vocabulary; it is not the repository's older canonical encoding with Bullet Bill channels or merged aliases.

Tiles are 16×16 pixels. Literal `#` and `X` use their respective sprites; `|` uses the stalk sprite. Mushroom platforms use context-dependent caps and centers. Pipes use their top and body sprites, handling the two body heights independently. A pipe whose top cell is `T` includes the plant, including a single-column pipe.

After all tiles and larger objects are drawn, the renderer places a black X inside every visited cell. At native resolution its stroke is two pixels wide with a three-pixel inset from the cell boundary. It then applies integer nearest-neighbor scaling, preserving the pixel art and overlay alignment. The X marks path occupancy, without drawing lines between cells or reconstructing trajectory order.

A supplied `(22, 16, 192)` level renders at **3072×256 pixels** with `scale=1`; at `scale=2` it becomes 6144×512. The complete 192-column stored grid is rendered, including the starting segment. Some original repository preview PNGs are only 2816 pixels wide; their narrower width should not be used to crop the encoding.

## Reuse from Python

The rendering functions return Pillow RGB images, so they can be used directly in a model's generation or evaluation code. The batch function returns PNG counts per split:

| Function | Purpose |
|---|---|
| `decode_encoding(encoding, *, channels_path=DEFAULT_CHANNELS)` | Return a Unicode character grid of shape `(H, W)` and a Boolean path mask of the same shape |
| `level_to_image(tile_grid, path_mask=None, *, assets_dir=DEFAULT_ASSETS, scale=1)` | Render an already decoded tile grid, with an optional path overlay |
| `encoding_to_image(encoding, *, channels_path=DEFAULT_CHANNELS, assets_dir=DEFAULT_ASSETS, show_path=True, scale=1)` | Decode and render an in-memory 21- or 22-channel array |
| `render_encoding_file(input_path, output_path, **options)` | Load a `.npy` encoding, render it, save a PNG, and return the image |
| `render_dataset(dataset=DEFAULT_DATASET, *, assets_dir=DEFAULT_ASSETS, scale=1)` | Save a sibling PNG for every encoding in all three splits and return a dictionary of PNG counts per split |

For an in-memory prediction that already satisfies the binary encoding rules:

```python
import numpy as np
from encoding_to_level_image import decode_encoding, encoding_to_image, level_to_image

# A small valid tile-only example: empty cells above a solid bottom row.
tile_only = np.zeros((21, 4, 12), dtype=np.uint8)
tile_only[2] = 1                      # empty tile channel
tile_only[2, -1, :] = 0
tile_only[0, -1, :] = 1              # solid tile channel
tile_image = encoding_to_image(tile_only)

# Add a separate path channel without replacing any tile bit.
path = np.zeros((1, 4, 12), dtype=np.uint8)
path[0, 2, 1:11] = 1
with_path = np.concatenate([tile_only, path], axis=0)
path_image = encoding_to_image(with_path, scale=2)
without_path_image = encoding_to_image(with_path, show_path=False)

tile_grid, path_mask = decode_encoding(with_path)
assert tile_grid.shape == path_mask.shape == (4, 12)
assert tile_grid.dtype.kind == "U" and path_mask.dtype == np.bool_
assert path_mask.sum() == 10
decoded_image = level_to_image(tile_grid, path_mask)
```

Model probabilities or logits need an explicit tile-selection step before calling this strict decoder. Apply that decision to the tile channels and decide the path mask separately; treating the path bit as another mutually exclusive tile class loses the intended overlap.

Load and render an existing sample, or save the returned Pillow image yourself:

```python
from pathlib import Path
import numpy as np
from encoding_to_level_image import encoding_to_image, render_encoding_file

source = Path(
    "preprocessed_dataset/train_set/"
    "exp1_0045__lvl195_over_lvl104/lvl195.npy"
)
array = np.load(source, allow_pickle=False)
image = encoding_to_image(array)
image.save("my-level-preview.png")

# The file helper performs loading and PNG saving in one call.
saved_image = render_encoding_file(source, "my-level-preview-large.png", scale=2)
tiles_only_image = encoding_to_image(array, show_path=False)
```

## Three training examples

Each embedded image links to its full-resolution PNG. All six supplied arrays have shape `(22, 16, 192)` and all example images use `scale=1` with the path overlay enabled. “Visited cells” counts the active path bits, not the number of recorded position samples.

### Training 1: Killer-74

Participant: `e8fa1eee-54e3-49af-bd53-33de244a87be`. Visited cells: **455**.

Source array: [`Killer-74.npy`](preprocessed_dataset/train_set/exp2_0077_pair1__Runner-31_over_Killer-74/Killer-74.npy).

[![Training sample 1: Killer-74 with black X path markers](encoding_level_images/train_set/01_Killer-74_e8fa1eee.png)](encoding_level_images/train_set/01_Killer-74_e8fa1eee.png)

### Training 2: lvl195

Participant: `2a8b8b1b-a314-4c4d-9721-8b94797932fb`. Visited cells: **474**.

Source array: [`lvl195.npy`](preprocessed_dataset/train_set/exp1_0045__lvl195_over_lvl104/lvl195.npy).

[![Training sample 2: lvl195 with black X path markers](encoding_level_images/train_set/02_lvl195_2a8b8b1b.png)](encoding_level_images/train_set/02_lvl195_2a8b8b1b.png)

### Training 3: Collector-42

Participant: `0cbaa3ae-5376-4d7f-8053-7daba92be456`. Visited cells: **263**.

Source array: [`Collector-42.npy`](preprocessed_dataset/train_set/exp2_0001_pair2__Runner-16_over_Collector-42/Collector-42.npy).

[![Training sample 3: Collector-42 with black X path markers](encoding_level_images/train_set/03_Collector-42_0cbaa3ae.png)](encoding_level_images/train_set/03_Collector-42_0cbaa3ae.png)

## Three test examples

### Test 1: Killer-94

Participant: `70d6994c-b6d4-4e92-9d50-e1c25931d401`. Visited cells: **336**.

Source array: [`Killer-94.npy`](preprocessed_dataset/test_set/exp2_0007_pair1__Collector-66_over_Killer-94/Killer-94.npy).

[![Test sample 1: Killer-94 with black X path markers](encoding_level_images/test_set/01_Killer-94_70d6994c.png)](encoding_level_images/test_set/01_Killer-94_70d6994c.png)

### Test 2: Runner-57

Participant: `6d710075-57ba-4545-bc65-e2a274f9f92b`. Visited cells: **336**.

Source array: [`Runner-57.npy`](preprocessed_dataset/test_set/exp2_0082_pair2__Killer-95_over_Runner-57/Runner-57.npy).

[![Test sample 2: Runner-57 with black X path markers](encoding_level_images/test_set/02_Runner-57_6d710075.png)](encoding_level_images/test_set/02_Runner-57_6d710075.png)

### Test 3: lvl40

Participant: `56f050c3-595e-40c7-a656-21076e66917f`. Visited cells: **446**.

Source array: [`lvl40.npy`](preprocessed_dataset/test_set/exp1_0124__lvl28_over_lvl40/lvl40.npy).

[![Test sample 3: lvl40 with black X path markers](encoding_level_images/test_set/03_lvl40_56f050c3.png)](encoding_level_images/test_set/03_lvl40_56f050c3.png)

## Traceability and interpretation

[`encoding_level_images/samples.json`](encoding_level_images/samples.json) records the selection seed, input and output paths, source hashes, encoding shapes, and visited-cell counts. Each sample's `encoding` path is relative to `preprocessed_dataset/`, and its `image` path is relative to `encoding_level_images/`. The record includes `encoding_sha256` for each input plus `channels_sha256` and `renderer_sha256` for the codebook and renderer. The source array links above identify the exact participant-specific observation rendered, including when that level appears elsewhere in the dataset.

A black X shows that the supplied path mask visited that cell. It does not show visit order, dwell time, or a guarantee that a generated path is playable. The picture shows static initial tile identities; collected coins, defeated enemies, and other evolving game state are not reconstructed from this encoding. For generated levels, use the same functions with your own valid arrays and assess path executability separately through the game dynamics.

## Verify the reusable functions

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The seven tests cover exact one-hot/multi-hot decoding, black X markers above sprites, unchanged unvisited cells, optional path hiding, nearest-neighbor scaling, literal stalks, unequal pipe-column heights, invalid encodings, and PNG saving without replacing the input array. The six displayed samples were also checked against their source tile/path text: all **2,310 visited cells** have X markers and all unvisited cells match the corresponding tile-only render.
