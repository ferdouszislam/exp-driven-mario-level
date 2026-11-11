#!/usr/bin/env python3
"""Render one/multi-hot Mario arrays as PNGs; see ENCODING_TO_LEVEL_IMAGE.md."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "preprocessed_dataset"
DEFAULT_CHANNELS = DEFAULT_DATASET / "channels.json"
DEFAULT_ASSETS = PROJECT_ROOT / "Mario-Fun-Study/smb/assets"
TILE_PIXELS = 16
BACKGROUND = (138, 165, 253)
SUPPORTED_TILES = set("XS-%tToQ@UL12gkrKRy#|")
PIPE_TILES = {"t", "T"}


def _channel_symbols(channels_path: str | Path) -> tuple[list[str], int | None]:
    schema = json.loads(Path(channels_path).read_text(encoding="utf-8"))
    if schema.get("axis_order", ["channel", "row", "column"]) != ["channel", "row", "column"]:
        raise ValueError("Only channel-first encodings (channel, row, column) are supported")
    entries = sorted(schema["channels"], key=lambda item: item["index"])
    if [entry["index"] for entry in entries] != list(range(len(entries))):
        raise ValueError("Channel indices must be consecutive starting at zero")
    path_index = schema.get("path_channel_index")
    if path_index is not None and (type(path_index) is not int or not 0 <= path_index < len(entries)):
        raise ValueError("Invalid path_channel_index in channel metadata")
    symbols = [entry["symbol"] for index, entry in enumerate(entries) if index != path_index]
    if len(set(symbols)) != len(symbols) or set(symbols) != SUPPORTED_TILES:
        raise ValueError("Expected the 21 literal tile symbols documented in channels.json")
    return symbols, path_index


def decode_encoding(encoding: np.ndarray, *, channels_path: str | Path = DEFAULT_CHANNELS
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return literal HxW tile characters and a boolean HxW path mask.

    Accept strict binary tile-only one-hot arrays (21,H,W) or merged tile/path
    arrays (22,H,W). Tile channels are decoded separately from the path channel.
    """
    symbols, path_index = _channel_symbols(channels_path)
    values = np.asarray(encoding)
    tile_count = len(symbols)
    if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] < 1:
        raise ValueError("Expected a nonempty channel-first array (C,H,W)")
    if values.dtype.kind not in "biuf" or not np.all((values == 0) | (values == 1)):
        raise ValueError("Encoding must contain only binary numeric values 0 and 1")
    if values.shape[0] == tile_count:
        tile_values = values
        path_mask = np.zeros(values.shape[1:], dtype=bool)
    elif path_index is not None and values.shape[0] == tile_count + 1:
        indices = [index for index in range(values.shape[0]) if index != path_index]
        tile_values = values[indices]
        path_mask = values[path_index].astype(bool, copy=True)
    else:
        raise ValueError(f"Expected {tile_count} tile-only or {tile_count + 1} tile/path channels")
    if not np.all(tile_values.sum(axis=0) == 1):
        raise ValueError("Each cell must have exactly one active tile channel")
    tiles = np.asarray(symbols, dtype="<U1")[tile_values.argmax(axis=0)]
    return tiles, path_mask


@lru_cache(maxsize=128)
def _sprite(assets_dir: str, name: str) -> Image.Image:
    with Image.open(Path(assets_dir) / f"{name}.png") as source:
        return source.convert("RGBA")


def level_to_image(tile_grid: np.ndarray, path_mask: np.ndarray | None = None, *,
                   assets_dir: str | Path = DEFAULT_ASSETS, scale: int = 1) -> Image.Image:
    """Render literal tiles, then black X path markers on top; return a Pillow RGB image.

    One source tile is 16x16 pixels. Scaling uses nearest-neighbor resampling.
    The original tile grid and path mask are never modified.
    """
    tiles = np.asarray(tile_grid)
    if tiles.ndim != 2 or min(tiles.shape) < 1:
        raise ValueError("tile_grid must be a nonempty HxW array")
    if any(not isinstance(value, str) or value not in SUPPORTED_TILES for value in tiles.flat):
        raise ValueError("tile_grid contains an unsupported tile character")
    if type(scale) is not int or scale < 1:
        raise ValueError("scale must be a positive integer")
    if path_mask is not None:
        path_mask = np.asarray(path_mask)
        if path_mask.shape != tiles.shape or not np.all((path_mask == 0) | (path_mask == 1)):
            raise ValueError("path_mask must be a binary HxW array matching tile_grid")
    height, width = tiles.shape
    assets = str(Path(assets_dir).resolve())
    image = Image.new("RGB", (width * TILE_PIXELS, height * TILE_PIXELS), BACKGROUND)

    def paste(name: str, column: float, row: float) -> None:
        sprite = _sprite(assets, name)
        image.paste(sprite, (round(column * TILE_PIXELS), round(row * TILE_PIXELS)), sprite)

    # Literal stalks are preserved; they are not regenerated under every platform.
    for row, column in zip(*np.where(tiles == "|")):
        paste("stalk", column, row)
    for row in range(height):
        column = 0
        while column < width:
            if tiles[row, column] != "%":
                column += 1
                continue
            end = column + 1
            while end < width and tiles[row, end] == "%":
                end += 1
            for current in range(column, end):
                name = "MS" if end - column == 1 else (
                    "ML" if current == column else "MR" if current == end - 1 else "MM")
                paste(name, current, row)
            column = end

    # Assemble pipes while respecting each column's actual cells and height.
    visited = set()
    for row in range(height):
        for column in range(width):
            if tiles[row, column] not in PIPE_TILES or (row, column) in visited:
                continue
            paired = (column + 1 < width and tiles[row, column + 1] in PIPE_TILES
                      and (row, column + 1) not in visited)
            columns = [column, column + 1] if paired else [column]
            if any(tiles[row, current] == "T" for current in columns):
                paste("chomper", column + (0.5 if paired else 0), row - 1)
            for side, current in enumerate(columns):
                current_row = row
                while (current_row < height and tiles[current_row, current] in PIPE_TILES
                       and (current_row, current) not in visited):
                    if paired:
                        name = ("TLP" if side == 0 else "TRP") if current_row == row else (
                            "[" if side == 0 else "]")
                    else:
                        name = "TSP" if current_row == row else "BSP"
                    paste(name, current, current_row)
                    visited.add((current_row, current))
                    current_row += 1

    for row in range(height):
        for column in range(width):
            tile = tiles[row, column]
            if tile in {"-", "%", "|", "t", "T"}:
                continue
            paste({"K": "wingk", "R": "wingr"}.get(tile, tile), column, row)

    # The path is a final layer, including over coins, enemies, pipes, and blocks.
    if path_mask is not None:
        draw = ImageDraw.Draw(image)
        for row, column in zip(*np.nonzero(path_mask)):
            left, top = int(column) * TILE_PIXELS + 3, int(row) * TILE_PIXELS + 3
            right, bottom = left + 9, top + 9
            draw.line((left, top, right, bottom), fill=(0, 0, 0), width=2)
            draw.line((left, bottom, right, top), fill=(0, 0, 0), width=2)
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    return image


def encoding_to_image(encoding: np.ndarray, *, channels_path: str | Path = DEFAULT_CHANNELS,
                      assets_dir: str | Path = DEFAULT_ASSETS, show_path: bool = True,
                      scale: int = 1) -> Image.Image:
    """Convert a one/multi-hot array directly to a reusable Pillow RGB image."""
    tiles, path_mask = decode_encoding(encoding, channels_path=channels_path)
    return level_to_image(tiles, path_mask if show_path else None, assets_dir=assets_dir, scale=scale)


def render_encoding_file(input_path: str | Path, output_path: str | Path, **options: Any) -> Image.Image:
    """Load an .npy encoding, save its rendered PNG, and return the Pillow image."""
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("The output PNG must not replace the input encoding")
    image = encoding_to_image(np.load(input_path, allow_pickle=False), **options)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")
    return image


def render_dataset(dataset: str | Path = DEFAULT_DATASET, *,
                   assets_dir: str | Path = DEFAULT_ASSETS, scale: int = 1) -> dict[str, int]:
    """Save a sibling PNG for every encoded level in the dataset's three splits."""
    dataset = Path(dataset).resolve()
    channels = dataset / "channels.json"
    counts = {}
    for split in ("train_set", "validation_set", "test_set"):
        sources = sorted((dataset / split).glob("*/*.npy"))
        for source in sources:
            render_encoding_file(source, source.with_suffix(".png"), channels_path=channels,
                                 assets_dir=assets_dir, scale=scale)
        counts[split] = len(sources)
    if not any(counts.values()):
        raise ValueError(f"No level encodings found in dataset splits: {dataset}")
    return counts


def render_examples(dataset: str | Path = DEFAULT_DATASET, *,
                    output_dir: str | Path = PROJECT_ROOT / "encoding_level_images",
                    count: int = 3, seed: int = 42, assets_dir: str | Path = DEFAULT_ASSETS,
                    scale: int = 1) -> dict[str, Any]:
    """Randomly choose unique playthroughs per split and save reproducible examples."""
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    dataset, output_dir = Path(dataset).resolve(), Path(output_dir).resolve()
    if output_dir == dataset or dataset in output_dir.parents:
        raise ValueError("Save example images outside the seven-file preprocessed dataset")
    manifest = json.loads((dataset / "manifest.json").read_text())
    channels = dataset / "channels.json"
    rng = random.Random(seed)
    samples, output_files = [], {}
    candidates_per_split = {}
    for split in ("train_set", "test_set"):
        candidates = {}
        for entry in sorted(manifest["instances"], key=lambda item: item["instance_id"]):
            if entry["split"] == split:
                for play in entry["playthroughs"]:
                    candidates.setdefault(play["playthrough_key"], (entry, play))
        keys = sorted(candidates)
        if count > len(keys):
            raise ValueError(f"Requested {count} samples from only {len(keys)} {split} playthroughs")
        candidates_per_split[split] = len(keys)
        for index, key in enumerate(rng.sample(keys, count), 1):
            entry, play = candidates[key]
            source = dataset / play["files"]["encoding"]
            encoding = np.load(source, allow_pickle=False)
            tiles, path_mask = decode_encoding(encoding, channels_path=channels)
            image = encoding_to_image(encoding, channels_path=channels, assets_dir=assets_dir, scale=scale)
            filename = f'{split}/{index:02d}_{play["level_id"]}_{entry["participant_id"][:8]}.png'
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            output_files[filename] = buffer.getvalue()
            samples.append({
                "split": split, "sample_number": index, "level_id": play["level_id"],
                "participant_id": entry["participant_id"], "playthrough_key": key,
                "instance_id": entry["instance_id"], "encoding": play["files"]["encoding"],
                "encoding_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "encoding_shape": list(encoding.shape), "image": filename,
                "image_size": list(image.size), "visited_cells": int(path_mask.sum()),
                "tile_shape": list(tiles.shape),
            })
    record = {"schema_version": 1, "seed": seed, "samples_per_split": count,
              "candidate_playthroughs_per_split": candidates_per_split,
              "selection": "first instance_id occurrence per playthrough; sorted keys; Random(seed).sample; train then test",
              "path_marker": "black X drawn after all tiles and objects", "scale": scale,
              "channels_sha256": hashlib.sha256(channels.read_bytes()).hexdigest(),
              "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "samples": samples}
    output_files["samples.json"] = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    if output_dir.exists():
        existing = {path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()}
        if existing != set(output_files):
            raise ValueError("Output directory contains a different sample selection; choose a new --output-dir")
    for filename, content in output_files.items():
        path = output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render", help="Render one .npy encoding to PNG")
    render.add_argument("input", type=Path)
    render.add_argument("output", type=Path)
    render.add_argument("--channels", type=Path, default=DEFAULT_CHANNELS)
    render.add_argument("--no-path", action="store_true")
    examples = commands.add_parser("examples", help="Render random unique playthroughs from each split")
    examples.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    examples.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "encoding_level_images")
    examples.add_argument("--count", type=int, default=3)
    examples.add_argument("--seed", type=int, default=42)
    dataset = commands.add_parser("dataset", help="Render every level beside its .npy in all splits")
    dataset.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    for command in (render, examples, dataset):
        command.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
        command.add_argument("--scale", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.command == "render":
            image = render_encoding_file(args.input, args.output, channels_path=args.channels,
                                         assets_dir=args.assets, show_path=not args.no_path, scale=args.scale)
            print(json.dumps({"output": str(args.output), "image_size": image.size}))
        elif args.command == "dataset":
            counts = render_dataset(args.dataset, assets_dir=args.assets, scale=args.scale)
            print(json.dumps({"dataset": str(args.dataset), "images": sum(counts.values()),
                              "images_per_split": counts}, indent=2))
        else:
            record = render_examples(args.dataset, output_dir=args.output_dir, count=args.count,
                                     seed=args.seed, assets_dir=args.assets, scale=args.scale)
            print(json.dumps({"output_dir": str(args.output_dir), "seed": args.seed,
                              "images": len(record["samples"]),
                              "samples_per_split": args.count}, indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Rendering failed: {error}\n")


if __name__ == "__main__":
    main()
