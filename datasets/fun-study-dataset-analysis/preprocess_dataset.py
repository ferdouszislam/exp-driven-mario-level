#!/usr/bin/env python3
"""Export reproducible Mario tile/path preference pairs; see PREPROCESSING.md."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import tempfile
from typing import Any
import uuid

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "Mario-Fun-Study/exp_data/survey_data/formal"
HEIGHT, WIDTH = 16, 192
TILE_SIZE, MARKER_Y_OFFSET = 16, 8
TILE_CHANNELS = (
    ("X", "solid"), ("S", "breakable_block"), ("-", "empty"),
    ("%", "mushroom_platform"), ("t", "normal_pipe"), ("T", "flower_pipe"),
    ("o", "coin"), ("Q", "coin_question_block"), ("@", "mushroom_question_block"),
    ("U", "mushroom_breakable_block"), ("L", "one_up_block"),
    ("1", "invisible_one_up_block"), ("2", "invisible_coin_block"),
    ("g", "goomba"), ("k", "green_koopa"), ("r", "red_koopa"),
    ("K", "flying_green_koopa"), ("R", "flying_red_koopa"), ("y", "spiky"),
    ("#", "solid_alias"), ("|", "mushroom_stalk"),
)
TILE_TO_ID = {symbol: index for index, (symbol, _) in enumerate(TILE_CHANNELS)}
PATH_CHANNEL = len(TILE_CHANNELS)
LEVEL_NAME = re.compile(r"(?:lvl\d+|(?:Runner|Killer|Collector)-\d+)")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_ids(participant: str, levels: list[str]) -> None:
    if str(uuid.UUID(participant)) != participant:
        raise ValueError(f"Noncanonical participant ID: {participant!r}")
    if not all(LEVEL_NAME.fullmatch(level) for level in levels):
        raise ValueError(f"Unexpected study level IDs: {levels!r}")
    if len(set(levels)) != len(levels):
        raise ValueError(f"A comparison repeats a level: {levels!r}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def annotation_metadata(row: dict[str, str], index: int, experiment: int,
                        levels: list[str]) -> dict[str, Any]:
    participant = row["ID"]
    validate_ids(participant, levels)
    # Identical submissions share this group even when their timestamps differ.
    reuse_key = json.dumps([experiment, participant, levels], separators=(",", ":"))
    return {
        "experiment": experiment,
        "participant_id": participant,
        "source_csv": "annotation.csv" if experiment == 1 else "annotation2.csv",
        "source_csv_line": index + 2,
        "source_csv_original_index": row.get("", ""),
        "source_annotation_id": f"exp{experiment}_row_{index + 1:04d}",
        "annotation_reuse_group": sha256(reuse_key.encode()),
        "annotation_timestamp": row["time"],
        "source_level_order": levels,
        "source_label": row["ANNT"] if experiment == 1 else "most_to_least_fun",
    }


def collect_pairs(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                        dict[str, int], list[dict[str, Any]]]:
    rows1, rows2 = read_csv(source / "annotation.csv"), read_csv(source / "annotation2.csv")
    pairs, excluded, ranking_annotations = [], [], []
    for index, row in enumerate(rows1):
        levels = ["lvl" + row["A"], "lvl" + row["B"]]
        metadata = annotation_metadata(row, index, 1, levels)
        label = row["ANNT"]
        if label in ("E", "N"):
            excluded.append({
                **metadata,
                "reason": "equally_fun" if label == "E" else "neither_fun",
                "playthrough_keys": [f'{row["ID"]}_{level}' for level in levels],
            })
            continue
        if label not in ("A", "B"):
            raise ValueError(f"Unknown Experiment I label: {label!r}")
        preferred, other = levels if label == "A" else levels[::-1]
        pairs.append({
            **metadata,
            "instance_id": f"exp1_{index + 1:04d}__{preferred}_over_{other}",
            "preferred_level_id": preferred,
            "less_preferred_level_id": other,
            "source_rank_positions": [0, 1] if label == "A" else [1, 0],
        })

    for index, row in enumerate(rows2):
        order = [row["R1"], row["R2"], row["R3"]]
        if sorted(order) != ["c", "k", "r"]:
            raise ValueError(f"Invalid three-level ranking at CSV line {index + 2}: {order}")
        by_code = {"r": row["R"], "k": row["K"], "c": row["C"]}
        levels = [by_code[code] for code in order]
        metadata = annotation_metadata(row, index, 2, levels)
        ranking_annotations.append(metadata)
        for pair_index, (first, second) in enumerate(((0, 1), (0, 2), (1, 2)), 1):
            preferred, other = levels[first], levels[second]
            pairs.append({
                **metadata,
                "instance_id": (f"exp2_{index + 1:04d}_pair{pair_index}__"
                                f"{preferred}_over_{other}"),
                "preferred_level_id": preferred,
                "less_preferred_level_id": other,
                "source_rank_positions": [first, second],
            })
    if len({pair["instance_id"] for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate generated instance folder IDs")
    return pairs, excluded, {"experiment_1_rows": len(rows1),
                             "experiment_2_ranking_rows": len(rows2)}, ranking_annotations


def assign_splits(pairs: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    """Keep connected annotations/playthroughs together, nearest to 80/20 by count."""
    parents = list(range(len(pairs)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners: dict[tuple[str, str], int] = {}
    for index, pair in enumerate(pairs):
        identities = [(field, pair[field]) for field in
                      ("source_annotation_id", "annotation_reuse_group")]
        identities.extend(("playthrough", f'{pair["participant_id"]}_{level}') for level in
                          (pair["preferred_level_id"], pair["less_preferred_level_id"]))
        for identity in identities:
            if identity in owners:
                first, second = find(index), find(owners[identity])
                parents[max(first, second)] = min(first, second)
            else:
                owners[identity] = index

    components: dict[int, list[int]] = {}
    for index in range(len(pairs)):
        components.setdefault(find(index), []).append(index)
    groups = []
    for indices in components.values():
        names = sorted(pairs[index]["instance_id"] for index in indices)
        group_id = "group_" + sha256(json_bytes(names))
        groups.append((group_id, indices))
    groups.sort(key=lambda group: sha256(f"{seed}:{group[0]}".encode()))

    train_target = len(pairs) * 4 // 5
    test_target = len(pairs) - train_target
    # Subset sum preserves whole groups and achieves the exact target when feasible.
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for group_index, (_, indices) in enumerate(groups):
        for count, selected in list(reachable.items()):
            next_count = count + len(indices)
            if next_count not in reachable:
                reachable[next_count] = selected + (group_index,)
    feasible = [count for count in reachable if 0 < count < len(pairs)]
    if not feasible:
        raise ValueError("Cannot form nonempty train/test sets without splitting reused data")
    test_count = min(feasible, key=lambda count: (abs(count - test_target),
                                                count > test_target, count))
    test_groups = set(reachable[test_count])
    for group_index, (group_id, indices) in enumerate(groups):
        split = "test_set" if group_index in test_groups else "train_set"
        for index in indices:
            pairs[index].update(split=split, split_group_id=group_id,
                                relative_directory=f'{split}/{pairs[index]["instance_id"]}')

    identities_by_split: dict[str, dict[str, set[str]]] = {}
    instance_counts, experiment_counts = {}, {}
    for split in ("train_set", "test_set"):
        items = [pair for pair in pairs if pair["split"] == split]
        instance_counts[split] = len(items)
        experiment_counts[split] = {str(experiment): sum(
            item["experiment"] == experiment for item in items) for experiment in (1, 2)}
        identities_by_split[split] = {
            "source_annotation_ids": {item["source_annotation_id"] for item in items},
            "annotation_reuse_groups": {item["annotation_reuse_group"] for item in items},
            "participant_ids": {item["participant_id"] for item in items},
            "level_ids": {item[field] for item in items for field in
                          ("preferred_level_id", "less_preferred_level_id")},
            "playthrough_keys": {f'{item["participant_id"]}_{item[field]}' for item in items
                                 for field in ("preferred_level_id", "less_preferred_level_id")},
        }
    shared = {field: len(identities_by_split["train_set"][field]
                        & identities_by_split["test_set"][field])
              for field in identities_by_split["train_set"]}
    if any(shared[field] for field in ("source_annotation_ids", "annotation_reuse_groups",
                                       "playthrough_keys")):
        raise ValueError("Related annotations or playthroughs cross train/test sets")
    return {
        "seed": seed, "requested_train_fraction": 0.8, "requested_test_fraction": 0.2,
        "target_train_instances": train_target, "target_test_instances": test_target,
        "instances": instance_counts, "by_experiment": experiment_counts,
        "actual_train_fraction": instance_counts["train_set"] / len(pairs),
        "actual_test_fraction": instance_counts["test_set"] / len(pairs),
        "connected_groups": len(groups), "train_groups": len(groups) - len(test_groups),
        "test_groups": len(test_groups), "shared_between_sets": shared,
        "unique_playthroughs": {split: len(fields["playthrough_keys"])
                                 for split, fields in identities_by_split.items()},
        "assignment_algorithm": "SHA256(seed:group_id) order; whole-group subset sum nearest test target",
        "group_by": ["source_annotation_id", "annotation_reuse_group", "playthrough_key"],
        "held_out_unit": "playthrough; participants and map geometries may recur across sets",
    }


def split_training_groups(entries: list[dict], fraction: float, seed: int) -> tuple[list, list]:
    """Use train.py's seeded whole-group validation split, preserving input order."""
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("Validation fraction must be finite and in [0, 1)")
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        groups.setdefault(entry["split_group_id"], []).append(entry)
    if fraction == 0 or len(groups) < 2:
        return entries, []
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    target = max(1, round(len(entries) * fraction))
    validation_ids, count = set(), 0
    for group_id in group_ids[:-1]:  # Leave at least one whole training group.
        if count >= target:
            break
        validation_ids.add(group_id)
        count += len(groups[group_id])
    fitting = [entry for entry in entries if entry["split_group_id"] not in validation_ids]
    validation = [entry for entry in entries if entry["split_group_id"] in validation_ids]
    return fitting, validation


def assign_validation_split(pairs: list[dict[str, Any]], summary: dict[str, Any],
                            fraction: float, seed: int) -> None:
    """Split only the original training pool; its connected groups stay intact."""
    original_train = [pair for pair in pairs if pair["split"] == "train_set"]
    _, validation = split_training_groups(original_train, fraction, seed)
    for pair in validation:
        pair.update(split="validation_set",
                    relative_directory=f'validation_set/{pair["instance_id"]}')

    # Retain the original train/test audit and report the final three-way layout.
    summary["original_train_test_split"] = json.loads(json.dumps(summary))
    summary["validation"] = {
        "seed": seed,
        "requested_fraction_of_original_train": fraction,
        "original_train_instances": len(original_train),
        "target_instances": max(1, round(len(original_train) * fraction)),
        "actual_fraction_of_original_train": len(validation) / len(original_train),
        "assignment_algorithm": (
            "train.py: sorted group IDs; random.Random(seed).shuffle; "
            "take whole groups until validation target, leaving one training group"
        ),
    }
    summary["requested_train_fraction"] *= 1 - fraction
    summary["requested_validation_fraction"] = 0.8 * fraction
    summary["target_validation_instances"] = summary["validation"]["target_instances"]
    summary["target_train_instances"] = len(original_train) - summary["target_validation_instances"]
    identities = {}
    for split in ("train_set", "validation_set", "test_set"):
        items = [pair for pair in pairs if pair["split"] == split]
        summary["instances"][split] = len(items)
        summary["by_experiment"][split] = {
            str(experiment): sum(item["experiment"] == experiment for item in items)
            for experiment in (1, 2)
        }
        label = split.removesuffix("_set")
        summary[f"actual_{label}_fraction"] = len(items) / len(pairs)
        summary[f"{label}_groups"] = len({item["split_group_id"] for item in items})
        identities[split] = {
            "split_group_ids": {item["split_group_id"] for item in items},
            "source_annotation_ids": {item["source_annotation_id"] for item in items},
            "annotation_reuse_groups": {item["annotation_reuse_group"] for item in items},
            "participant_ids": {item["participant_id"] for item in items},
            "level_ids": {item[field] for item in items for field in
                          ("preferred_level_id", "less_preferred_level_id")},
            "playthrough_keys": {f'{item["participant_id"]}_{item[field]}' for item in items
                                 for field in ("preferred_level_id", "less_preferred_level_id")},
        }
        summary["unique_playthroughs"][split] = len(identities[split]["playthrough_keys"])
    shared = {}
    for first, second in (("train_set", "validation_set"), ("train_set", "test_set"),
                          ("validation_set", "test_set")):
        counts = {field: len(values & identities[second][field])
                  for field, values in identities[first].items()}
        if any(counts[field] for field in ("split_group_ids", "source_annotation_ids",
                                           "annotation_reuse_groups", "playthrough_keys")):
            raise ValueError(f"Related annotations or playthroughs cross {first}/{second}")
        shared[f"{first}_vs_{second}"] = counts
    summary["shared_between_sets"] = shared


@dataclass
class Playthrough:
    level_id: str
    key: str
    tile_text: bytes
    path_text: bytes
    encoding: bytes
    stats: dict[str, Any]


def load_playthrough(source: Path, participant: str, level: str,
                     hashes: dict[str, Any]) -> Playthrough:
    key = f"{participant}_{level}"
    level_relative, result_relative = f"levels/{level}.lvl", f"res/{key}.json"
    for relative in (level_relative, result_relative):
        content = (source / relative).read_bytes()
        hashes[relative] = {"sha256": sha256(content), "bytes": len(content)}

    rows = (source / level_relative).read_text(encoding="utf-8").splitlines()
    if len(rows) != HEIGHT or any(len(row) != WIDTH for row in rows):
        raise ValueError(f"Expected a {HEIGHT}x{WIDTH} grid: {level_relative}")
    unknown = set("".join(rows)) - TILE_TO_ID.keys()
    if unknown:
        raise ValueError(f"Unknown/omitted tile characters in {level}: {sorted(unknown)}")
    tiles = np.array([[TILE_TO_ID[symbol] for symbol in row] for row in rows])
    tile_channels = np.eye(PATH_CHANNEL, dtype=np.uint8)[tiles].transpose(2, 0, 1)

    result = json.loads((source / result_relative).read_bytes())
    xy = np.asarray(result["trace"], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or not np.isfinite(xy).all():
        raise ValueError(f"Expected a nonempty finite Nx2 trace: {result_relative}")
    columns = np.floor(xy[:, 0] / TILE_SIZE)
    grid_rows = np.floor((xy[:, 1] - MARKER_Y_OFFSET) / TILE_SIZE)
    valid = ((grid_rows >= 0) & (grid_rows < HEIGHT)
             & (columns >= 0) & (columns < WIDTH))
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[grid_rows[valid].astype(int), columns[valid].astype(int)] = 1
    tensor = np.concatenate([tile_channels, mask[None, :, :]], axis=0)
    if not np.all(tensor[:-1].sum(axis=0) == 1):
        raise ValueError(f"Invalid tile one-hot representation: {key}")
    if not np.array_equal(tensor.sum(axis=0), 1 + mask):
        raise ValueError(f"Invalid merged tile/path representation: {key}")
    buffer = io.BytesIO()
    np.save(buffer, tensor, allow_pickle=False)
    path_text = "\n".join("".join("P" if cell else "x" for cell in row) for row in mask) + "\n"
    completion = float(result["completing-ratio"])
    if not math.isfinite(completion):
        raise ValueError(f"Nonfinite completion ratio: {key}")
    return Playthrough(
        level_id=level, key=key,
        tile_text=("\n".join(rows) + "\n").encode(),
        path_text=path_text.encode(), encoding=buffer.getvalue(),
        stats={
            "ordered_samples": len(xy),
            "valid_samples": int(valid.sum()),
            "out_of_grid_samples": int((~valid).sum()),
            "above_grid_samples": int((grid_rows < 0).sum()),
            "below_grid_samples": int((grid_rows >= HEIGHT).sum()),
            "horizontal_out_of_grid_samples": int(((columns < 0) | (columns >= WIDTH)).sum()),
            "visited_cells": int(mask.sum()),
            "status": result["status"],
            "completing_ratio": completion,
        },
    )


def build_output(source: Path, split_seed: int = 42, validation_fraction: float = 0.2
                 ) -> tuple[dict[str, bytes], dict[str, Any]]:
    pairs, excluded, input_counts, ranking_annotations = collect_pairs(source)
    split_summary = assign_splits(pairs, split_seed)
    if not math.isfinite(validation_fraction) or not 0 <= validation_fraction < 1:
        raise ValueError("Validation fraction must be finite and in [0, 1)")
    if validation_fraction:
        assign_validation_split(pairs, split_summary, validation_fraction, split_seed)
    hashes: dict[str, Any] = {}
    for name in ("annotation.csv", "annotation2.csv"):
        data = (source / name).read_bytes()
        hashes[name] = {"sha256": sha256(data), "bytes": len(data)}
    cache: dict[str, Playthrough] = {}
    files: dict[str, bytes] = {}
    manifest = []
    for pair in pairs:
        folder = pair["relative_directory"]
        playthrough_entries = []
        for level in (pair["preferred_level_id"], pair["less_preferred_level_id"]):
            key = f'{pair["participant_id"]}_{level}'
            if key not in cache:
                cache[key] = load_playthrough(source, pair["participant_id"], level, hashes)
            play = cache[key]
            names = {"tiles": f"{folder}/{level}.tiles.txt",
                     "path": f"{folder}/{level}.path.txt",
                     "encoding": f"{folder}/{level}.npy"}
            files[names["tiles"]] = play.tile_text
            files[names["path"]] = play.path_text
            files[names["encoding"]] = play.encoding
            playthrough_entries.append({
                "level_id": level, "playthrough_key": key, "files": names,
                "source_level": f"levels/{level}.lvl",
                "source_result": f"res/{key}.json", "trace_stats": play.stats,
            })
        ranking_file = f"{folder}/ranking.txt"
        files[ranking_file] = (pair["preferred_level_id"] + "\n"
                               + pair["less_preferred_level_id"] + "\n").encode()
        manifest.append({**pair, "ranking_file": ranking_file,
                         "playthroughs": playthrough_entries})

    omitted_keys = {key for item in excluded for key in item["playthrough_keys"]}
    omitted_levels = {level for item in excluded for level in item["source_level_order"]}
    retained_levels = {play.level_id for play in cache.values()}
    reuse_counts = Counter(item["annotation_reuse_group"] for item in ranking_annotations)
    repeated_groups = sum(count > 1 for count in reuse_counts.values())
    additional_submissions = sum(count - 1 for count in reuse_counts.values())
    n_exp1 = sum(pair["experiment"] == 1 for pair in pairs)
    summary = {
        "schema_version": 3 if validation_fraction else 2,
        "input": input_counts,
        "split": split_summary,
        "output": {
            "pair_instances": len(pairs), "experiment_1_pairs": n_exp1,
            "experiment_2_pairs": len(pairs) - n_exp1,
            "files_per_instance": 7, "pair_side_appearances": 2 * len(pairs),
            "unique_playthroughs": len(cache), "unique_levels": len(retained_levels),
            "participants": len({pair["participant_id"] for pair in pairs}),
        },
        "excluded": {
            "equal_fun_comparisons": sum(item["source_label"] == "E" for item in excluded),
            "neither_fun_comparisons": sum(item["source_label"] == "N" for item in excluded),
            "total_comparisons": len(excluded), "playthrough_appearances": 2 * len(excluded),
            "unique_playthroughs": len(omitted_keys), "unique_levels": len(omitted_levels),
            "level_ids_also_retained_for_other_judgments": len(omitted_levels & retained_levels),
            "playthrough_keys_also_retained": len(omitted_keys & cache.keys()),
        },
        "preserved_repeated_rankings": {
            "repeated_submission_groups": repeated_groups,
            "additional_submissions": additional_submissions,
            "pairs_from_additional_submissions": 3 * additional_submissions,
        },
        "unique_playthrough_path_stats": {
            "ordered_samples": sum(play.stats["ordered_samples"] for play in cache.values()),
            "out_of_grid_samples": sum(play.stats["out_of_grid_samples"] for play in cache.values()),
            "playthroughs_with_out_of_grid_samples": sum(
                play.stats["out_of_grid_samples"] > 0 for play in cache.values()),
            "visited_cells": sum(play.stats["visited_cells"] for play in cache.values()),
            "statuses": dict(Counter(play.stats["status"] for play in cache.values())),
        },
        "policies": {
            "annotation_subset": "supplied cleaned CSVs; no additional progress/status filter",
            "experiment_1": "retain A/B; exclude E/N annotation rows",
            "experiment_2": "pairs (rank1,rank2), (rank1,rank3), (rank2,rank3)",
            "deduplicate_rankings": False,
            "map_crop": None,
            "path_grid": "floor((y-8)/16), floor(x/16); sampled occupancy; skip out-of-grid",
            "path_symbols": {"visited": "P", "unvisited": "x"},
            "tile_aliases": "preserve literal # and |; omit b and B; reject unknown symbols",
        },
        "reproduction": {
            "script_sha256": sha256(Path(__file__).read_bytes()),
            "numpy_version": np.__version__,
            "source_hashes_file": "source_hashes.json",
        },
    }
    channels = {
        "schema_version": 1,
        "shape": [PATH_CHANNEL + 1, HEIGHT, WIDTH], "dtype": "uint8",
        "axis_order": ["channel", "row", "column"],
        "channels": [{"index": index, "symbol": symbol, "name": name}
                     for index, (symbol, name) in enumerate(TILE_CHANNELS)]
                    + [{"index": PATH_CHANNEL, "symbol": "P", "name": "PATH"}],
        "path_channel_index": PATH_CHANNEL,
        "path_text_symbols": {"visited": "P", "unvisited": "x"},
        "active_bits_per_cell": "1 for unvisited cells; 2 for visited cells",
    }
    files["channels.json"] = json_bytes(channels)
    files["manifest.json"] = json_bytes({"schema_version": 2, "instances": manifest})
    files["summary.json"] = json_bytes(summary)
    files["excluded_comparisons.json"] = json_bytes({"comparisons": excluded})
    files["source_hashes.json"] = json_bytes({"algorithm": "sha256", "files": hashes})
    return files, summary


def verify_output(output: Path, expected: dict[str, bytes]) -> str:
    """Compare the entire output tree to a fresh reconstruction from the sources."""
    if not output.is_dir() or output.is_symlink():
        raise ValueError(f"Expected a regular output directory: {output}")
    paths = list(output.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Output contains a symlink")
    actual_files = {path.relative_to(output).as_posix() for path in paths if path.is_file()}
    expected_dirs = {parent.as_posix() for name in expected for parent in Path(name).parents
                     if parent != Path(".")}
    actual_dirs = {path.relative_to(output).as_posix() for path in paths if path.is_dir()}
    if actual_files != set(expected) or actual_dirs != expected_dirs:
        missing, extra = sorted(set(expected) - actual_files), sorted(actual_files - set(expected))
        raise ValueError(f"Output tree differs; missing={missing[:5]}, extra={extra[:5]}; "
                         f"unexpected/missing directories={sorted(actual_dirs ^ expected_dirs)[:5]}")
    digest = hashlib.sha256()
    for name in sorted(expected):
        content = (output / name).read_bytes()
        if content != expected[name]:
            raise ValueError(f"Output differs from source reconstruction: {name}")
        if name.endswith(".npy"):
            tensor = np.load(io.BytesIO(content), allow_pickle=False)
            if tensor.shape != (22, HEIGHT, WIDTH) or tensor.dtype != np.uint8:
                raise ValueError(f"Invalid tensor shape/dtype: {name}")
            if not np.all((tensor == 0) | (tensor == 1)):
                raise ValueError(f"Nonbinary tensor: {name}")
            if not np.all(tensor[:-1].sum(axis=0) == 1):
                raise ValueError(f"Invalid tile channels: {name}")
            if not np.array_equal(tensor.sum(axis=0), 1 + tensor[-1]):
                raise ValueError(f"Invalid path overlap: {name}")
        digest.update(name.encode() + b"\0" + hashlib.sha256(content).digest())
    return digest.hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def migrate_output(output: Path, expected: dict[str, bytes], legacy: dict[str, bytes]) -> str:
    """Validate a flat or train/test export before moving its comparison folders."""
    manifest = json.loads(expected["manifest.json"])
    legacy = dict(legacy)
    legacy_summary = json.loads(legacy["summary.json"])
    if not (output / "train_set").exists():
        legacy = {name.split("/", 1)[1] if name.startswith(("train_set/", "test_set/"))
                  else name: content for name, content in legacy.items()}
        legacy_manifest = json.loads(legacy["manifest.json"])
        legacy_manifest["schema_version"] = 1
        for item in legacy_manifest["instances"]:
            for field in ("split", "split_group_id", "relative_directory"):
                item.pop(field)
            item["ranking_file"] = item["ranking_file"].split("/", 1)[1]
            for play in item["playthroughs"]:
                play["files"] = {field: name.split("/", 1)[1]
                                 for field, name in play["files"].items()}
        legacy["manifest.json"] = json_bytes(legacy_manifest)
        legacy_summary.pop("split")
        legacy_summary["schema_version"] = 1
    original_summary = json.loads((output / "summary.json").read_bytes())
    legacy_summary["reproduction"]["script_sha256"] = original_summary["reproduction"]["script_sha256"]
    legacy["summary.json"] = json_bytes(legacy_summary)
    verify_output(output, legacy)

    originals = {name: (output / name).read_bytes() for name in ("manifest.json", "summary.json")}
    moved = []
    created = []
    try:
        for split in sorted({item["split"] for item in manifest["instances"]}):
            directory = output / split
            if not directory.exists():
                directory.mkdir()
                created.append(directory)
        old_manifest = json.loads(originals["manifest.json"])
        old_directories = {item["instance_id"]: item.get("relative_directory", item["instance_id"])
                           for item in old_manifest["instances"]}
        for item in manifest["instances"]:
            old = output / old_directories[item["instance_id"]]
            new = output / item["relative_directory"]
            if old == new:
                continue
            old.rename(new)
            moved.append((old, new))
        for name in originals:
            atomic_write(output / name, expected[name])
        return verify_output(output, expected)
    except BaseException:
        for name, content in originals.items():
            atomic_write(output / name, content)
        for old, new in reversed(moved):
            new.rename(old)
        for directory in reversed(created):
            directory.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Formal study directory with levels/, res/, and cleaned CSVs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "preprocessed_dataset",
                        help="Superfolder for pair folders and global JSON metadata")
    parser.add_argument("--verify-only", action="store_true",
                        help="Reconstruct expected files and verify an existing output, without writes")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Seed for reproducible grouped 80/20 assignment (default: 42)")
    parser.add_argument("--validation-fraction", type=float, default=0.2,
                        help="Fraction of train_set held out in whole groups (default: 0.2; 0 disables)")
    parser.add_argument("--migrate-layout", action="store_true",
                        help="Validate and move an existing flat or train/test export into the new splits")
    args = parser.parse_args()
    if args.verify_only and args.migrate_layout:
        parser.error("--verify-only cannot be combined with --migrate-layout")
    source, output = args.source.resolve(), args.output.absolute()
    if output == source or source in output.parents or output in source.parents:
        parser.error("Output and source must be separate, non-nested directories")
    try:
        files, summary = build_output(source, args.split_seed, args.validation_fraction)
        if output.exists():
            if args.migrate_layout and not (output / "validation_set").exists():
                legacy, _ = build_output(source, args.split_seed, validation_fraction=0)
                digest = migrate_output(output, files, legacy)
                action = "migrated_and_verified"
            else:
                digest = verify_output(output, files)
                action = "verified_existing"
        elif args.verify_only:
            raise ValueError(f"Output does not exist: {output}")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            # A failed generation/verification cannot leave a partial target dataset.
            with tempfile.TemporaryDirectory(prefix=f".{output.name}.build-", dir=output.parent) as temp:
                staging = Path(temp) / "dataset"
                staging.mkdir()
                for name, content in files.items():
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                digest = verify_output(staging, files)
                staging.rename(output)
            action = "created_and_verified"
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Preprocessing failed: {error}\n"
                       "Existing output is never overwritten; use a new --output path for changed inputs.\n")
    print(json.dumps({"action": action, "output": str(output),
                      "pair_instances": summary["output"]["pair_instances"],
                      "train_instances": summary["split"]["instances"]["train_set"],
                      "validation_instances": summary["split"]["instances"].get("validation_set", 0),
                      "test_instances": summary["split"]["instances"]["test_set"],
                      "split_seed": args.split_seed,
                      "validation_fraction": args.validation_fraction,
                      "verified_files": len(files), "dataset_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
