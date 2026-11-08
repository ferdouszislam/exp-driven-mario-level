"""Load only the prepared training comparisons, preserving annotation groups."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


INPUT_SHAPE = (22, 16, 192)
TILE_SYMBOLS = ["X", "S", "-", "%", "t", "T", "o", "Q", "@", "U", "L", "1", "2", "g", "k", "r", "K", "R", "y", "#", "|"]


@dataclass
class TrainingData:
    entries: list[dict]
    tensors: dict[str, Tensor]
    channels: dict
    available_comparisons: int
    dataset_fingerprint: str
    selection: dict


def _training_path(root: Path, relative_path: str) -> Path:
    """Reject paths that could resolve outside this root's train_set directory."""
    if not isinstance(relative_path, str):
        raise ValueError("A training file path must be a string")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "train_set":
        raise ValueError(f"Not a train_set path: {relative_path!r}")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root / "train_set"):
        raise ValueError(f"Training path resolves outside train_set: {relative_path}")
    return target


def _hash_part(digest, name: str, contents: bytes) -> None:
    """Length-prefix both names and contents to make hash boundaries explicit."""
    name_bytes = name.encode("utf-8")
    for part in (name_bytes, contents):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)


def _check_channels(channels: dict) -> None:
    saved = channels.get("channels", [])
    expected_symbols = TILE_SYMBOLS + ["P"]
    if (
        channels.get("shape") != list(INPUT_SHAPE)
        or channels.get("dtype") != "uint8"
        or channels.get("axis_order") != ["channel", "row", "column"]
        or channels.get("path_channel_index") != 21
        or len(saved) != 22
        or [channel.get("index") for channel in saved] != list(range(22))
        or [channel.get("symbol") for channel in saved] != expected_symbols
        or saved[21].get("name") != "PATH"
    ):
        raise ValueError("channels.json does not describe the specified 22-channel tile/path encoding")


def load_training_data(root: Path, max_samples: int | None, seed: int) -> TrainingData:
    """Validate training files and select at most max_samples whole comparisons.

    Selection never breaks a split_group_id. A cap may therefore be underfilled.
    All training arrays are validated; only selected arrays are cached as float32.
    No path or array belonging to the test split is opened or hashed.
    """
    root = Path(root).resolve()
    train_root = root / "train_set"
    if not train_root.is_dir() or train_root.resolve() != train_root:
        raise ValueError(f"Expected an ordinary train_set directory inside {root}")
    if max_samples is not None and (isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1):
        raise ValueError("max_samples must be a positive integer or None")

    channels_bytes = (root / "channels.json").read_bytes()
    channels = json.loads(channels_bytes)
    _check_channels(channels)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("Expected manifest schema_version 2 with explicit split assignments")
    entries = [entry for entry in manifest["instances"] if entry.get("split") == "train_set"]
    if not entries:
        raise ValueError("No train_set comparisons in manifest.json")
    entries.sort(key=lambda entry: entry["instance_id"])
    ids = [entry["instance_id"] for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate train_set instance IDs in manifest.json")
    actual_ids = {directory.name for directory in train_root.iterdir() if directory.is_dir()}
    if set(ids) != actual_ids:
        raise ValueError("Train-set comparison directories do not match the train manifest entries")
    if max_samples is not None and max_samples > len(entries):
        raise ValueError(f"Requested {max_samples} comparisons, but train_set contains {len(entries)}")

    groups: dict[str, list[dict]] = {}
    for entry in entries:
        group_id = entry.get("split_group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"Missing split_group_id for {entry['instance_id']}")
        groups.setdefault(group_id, []).append(entry)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    selected: list[dict] = []
    for group_id in group_ids:
        group = groups[group_id]
        if max_samples is None or len(selected) + len(group) <= max_samples:
            selected.extend(group)
    selected.sort(key=lambda entry: entry["instance_id"])
    if not selected:
        raise ValueError("The sample cap is too small to fit any complete split_group_id")
    selected_ids = {entry["instance_id"] for entry in selected}

    digest = hashlib.sha256()
    _hash_part(digest, "channels.json", channels_bytes)
    tensors: dict[str, Tensor] = {}
    for entry in entries:
        instance_id = entry["instance_id"]
        directory_name = f"train_set/{instance_id}"
        if PurePosixPath(instance_id).name != instance_id or instance_id in (".", ".."):
            raise ValueError(f"Invalid instance_id: {instance_id!r}")
        if entry.get("relative_directory") != directory_name:
            raise ValueError(f"Unexpected comparison directory for {instance_id}")
        ranking_name = f"{directory_name}/ranking.txt"
        if entry.get("ranking_file") != ranking_name:
            raise ValueError(f"Unexpected ranking file for {instance_id}")
        ranking_bytes = _training_path(root, ranking_name).read_bytes()
        ranking = ranking_bytes.decode("utf-8").splitlines()
        expected_ranking = [entry["preferred_level_id"], entry["less_preferred_level_id"]]
        if ranking != expected_ranking or len(set(ranking)) != 2:
            raise ValueError(f"Ranking and preferred/other metadata disagree for {instance_id}")
        plays = entry.get("playthroughs", [])
        if len(plays) != 2 or {play["level_id"] for play in plays} != set(ranking):
            raise ValueError(f"Playthrough IDs do not match ranking for {instance_id}")
        if any(not isinstance(level_id, str) or PurePosixPath(level_id).name != level_id or level_id in (".", "..") for level_id in ranking):
            raise ValueError(f"Invalid level IDs for {instance_id}")
        selected_entry = instance_id in selected_ids
        if selected_entry:
            metadata = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            _hash_part(digest, f"{directory_name}/manifest_entry", metadata)
            _hash_part(digest, ranking_name, ranking_bytes)
        for play in sorted(plays, key=lambda play: play["level_id"]):
            encoding_name = f"{directory_name}/{play['level_id']}.npy"
            if play["files"]["encoding"] != encoding_name:
                raise ValueError(f"Unexpected encoding path for {instance_id}/{play['level_id']}")
            encoding_bytes = _training_path(root, encoding_name).read_bytes()
            array = np.load(io.BytesIO(encoding_bytes), allow_pickle=False)
            if array.shape != INPUT_SHAPE or array.dtype != np.uint8:
                raise ValueError(f"Expected uint8 {INPUT_SHAPE} array: {encoding_name}")
            if not np.all((array == 0) | (array == 1)) or not np.all(array[:21].sum(axis=0) == 1):
                raise ValueError(f"Expected binary multi-hot tiles plus independent path bit: {encoding_name}")
            if selected_entry:
                _hash_part(digest, encoding_name, encoding_bytes)
                tensors[encoding_name] = torch.from_numpy(array.astype(np.float32))

    return TrainingData(
        entries=selected,
        tensors=tensors,
        channels=channels,
        available_comparisons=len(entries),
        dataset_fingerprint=digest.hexdigest(),
        selection={
            "requested_max_samples": max_samples,
            "selected_comparisons": len(selected),
            "available_comparisons": len(entries),
            "selected_groups": len({entry["split_group_id"] for entry in selected}),
            "available_groups": len(groups),
            "seed": seed,
            "policy": "seeded whole-group greedy selection at or below the comparison cap",
        },
    )


class ComparisonDataset(Dataset):
    """Expose preferred-first comparisons or both class-balanced ordered views."""

    def __init__(self, data: TrainingData, entries: list[dict], ordered: bool) -> None:
        self.data = data
        self.entries = entries
        self.ordered = ordered
        selected_ids = {entry["instance_id"] for entry in data.entries}
        if any(entry.get("split") != "train_set" or entry["instance_id"] not in selected_ids for entry in entries):
            raise ValueError("ComparisonDataset accepts only the selected training entries")

    def __len__(self) -> int:
        return len(self.entries) * (2 if self.ordered else 1)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        entry = self.entries[index // 2 if self.ordered else index]
        reverse = self.ordered and index % 2 == 1
        plays = {play["level_id"]: play for play in entry["playthroughs"]}
        preferred = self.data.tensors[plays[entry["preferred_level_id"]]["files"]["encoding"]]
        other = self.data.tensors[plays[entry["less_preferred_level_id"]]["files"]["encoding"]]
        if reverse:
            return other, preferred, torch.tensor(0.0, dtype=torch.float32)
        return preferred, other, torch.tensor(1.0, dtype=torch.float32)
