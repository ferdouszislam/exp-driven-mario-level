"""Reconstruct train/validation levels and sample new levels from saved checkpoints.

With no arguments, process every checkpoints/*.pt under model-training.
Outputs live beside each checkpoints folder in reconstructions/{split}/{checkpoint}.
Use --output-dir to choose another output root, preserving relative run folders
when searching recursively.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import torch

from data import load_training_data
from reconstruction_previews import (
    save_checkpoint_reconstructions, save_generated_levels,
    select_preview_levels,
)


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE.parent / "datasets/fun-study-dataset-analysis/preprocessed_dataset"
SPLITS = ("train", "validation", "generated")


def discover_checkpoints(root: Path, *, checkpoints_only: bool = False) -> list[Path]:
    """Find every saved model in checkpoint folders, in a stable order."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Checkpoint search directory does not exist: {root}")
    if checkpoints_only or root.name == "checkpoints":
        return sorted(path for path in root.glob("*.pt") if path.is_file())
    return sorted(path for path in root.rglob("*.pt")
                  if path.parent.name == "checkpoints" and path.is_file())


@lru_cache(maxsize=1)
def available_models() -> dict:
    """Load local model definitions without importing any training entrypoint."""
    models = {}
    for path in sorted(HERE.glob("*/model.py")):
        spec = importlib.util.spec_from_file_location(f"mario_generation_{path.parent.name}", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load model definition: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metadata = module.architecture_metadata()
        version = metadata["architecture_version"]
        if version in models:
            raise ValueError(f"Duplicate architecture version: {version}")
        models[version] = module
    return models


def restore_model(checkpoint: dict, device: torch.device):
    """Require matching architecture metadata and all saved model parameters."""
    metadata = checkpoint["config"]["architecture"]
    version = metadata["architecture_version"]
    module = available_models().get(version)
    if module is None:
        raise ValueError(f"No local model definition matches checkpoint architecture {version!r}")
    if metadata != module.architecture_metadata():
        raise ValueError(f"Checkpoint architecture metadata differs from the local {version} definition")
    model = module.PairwiseCNNVAE()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


@lru_cache(maxsize=1)
def cached_data(dataset: Path, cap: int | None, seed: int):
    """Reuse validated tensors while processing checkpoints from the same run."""
    return load_training_data(dataset, cap, seed)


def checkpoint_inputs(checkpoint: dict, dataset_override: Path | None):
    """Recover the original fitting/validation memberships without making a new split."""
    config = checkpoint["config"]
    saved_dataset = Path(config["dataset"])
    dataset = (dataset_override if dataset_override is not None else
               saved_dataset if saved_dataset.is_dir() else DEFAULT_DATASET).resolve()
    data = cached_data(dataset, config.get("requested_sample_cap"), config["seed"])
    if data.channels != checkpoint["channels"]:
        raise ValueError("Dataset channels differ from the checkpoint codebook")
    if config.get("dataset_fingerprint") not in (None, data.dataset_fingerprint):
        raise ValueError(f"Dataset fingerprint differs from the checkpoint: {dataset}")
    selected_ids = {entry["instance_id"] for entry in data.entries}
    if "selected_instance_ids" in checkpoint and selected_ids != set(checkpoint["selected_instance_ids"]):
        raise ValueError("Dataset selection differs from the checkpoint's selected comparisons")
    entries = {entry["instance_id"]: entry for entry in data.entries + data.validation_entries}
    fitting_ids = checkpoint["fitting_instance_ids"]
    validation_ids = checkpoint["validation_instance_ids"]
    if not fitting_ids or set(fitting_ids) & set(validation_ids):
        raise ValueError("Checkpoint must contain nonempty, disjoint fitting and validation memberships")
    if not set(fitting_ids) <= selected_ids:
        raise ValueError("Checkpoint fitting memberships are outside its selected train_set")
    if not set(fitting_ids + validation_ids) <= entries.keys():
        raise ValueError("Checkpoint comparisons are missing from the selected dataset")
    fitting = [entries[instance] for instance in fitting_ids]
    validation = [entries[instance] for instance in validation_ids]
    if data.validation_entries and any(entry["split"] != "validation_set" for entry in validation):
        raise ValueError("Checkpoint validation memberships differ from the saved validation_set")
    fitting_plays = {play["playthrough_key"] for entry in fitting for play in entry["playthroughs"]}
    validation_plays = {play["playthrough_key"] for entry in validation for play in entry["playthroughs"]}
    if fitting_plays & validation_plays:
        raise ValueError("Checkpoint fitting and validation portions share playthroughs")
    return data, fitting, validation, dataset


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def publish_outputs(staging: Path, output: Path, checkpoint_name: str, index: list[dict]) -> None:
    """Publish all three portions and their index, restoring old files on failure."""
    write_json(staging / "index.json", index)
    backup = staging / "previous"
    backup.mkdir()
    installed, saved = [], []
    try:
        for split in SPLITS:
            destination = output / split / checkpoint_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                manifest = json.loads((destination / "samples.json").read_text(encoding="utf-8"))
                if manifest.get("split") != split:
                    raise ValueError(f"Cannot replace an unrecognized output directory: {destination}")
                destination.replace(backup / split)
                saved.append((backup / split, destination))
            (staging / split).replace(destination)
            installed.append((destination, staging / split))
        destination = output / "index.json"
        if destination.exists():
            destination.replace(backup / "index.json")
            saved.append((backup / "index.json", destination))
        (staging / "index.json").replace(destination)
        installed.append((destination, staging / "index.json"))
    except BaseException:
        for destination, original in reversed(installed):
            destination.replace(original)
        for original, destination in reversed(saved):
            original.replace(destination)
        raise


def process_checkpoint(path: Path, device: torch.device,
                       dataset_override: Path | None = None, seed: int | None = None,
                       output_dir: Path | None = None) -> dict:
    """Restore one saved model and write both reconstructions and prior samples."""
    path = Path(path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = restore_model(checkpoint, device)
    data, fitting, validation, dataset = checkpoint_inputs(checkpoint, dataset_override)
    config = checkpoint["config"]
    selection_seed = config["seed"] if seed is None else seed
    posterior_seed = (selection_seed + 2) % (2**32)
    generation_seed = (selection_seed + 3) % (2**32)
    metadata = config["architecture"]
    categorical = (metadata.get("tile_output_activation") == "softmax"
                   or "softmax" in metadata["architecture_version"])
    combination = path.parent.parent
    output = Path(output_dir).resolve() if output_dir is not None else combination / "reconstructions"
    checkpoint_relative = path.relative_to(combination).as_posix()
    epoch = checkpoint["epoch"]
    record = {"checkpoint": checkpoint_relative, "epoch": epoch,
              "architecture": metadata["architecture_version"], "dataset": str(dataset)}
    index_path = output / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    if not isinstance(index, list):
        raise ValueError(f"Expected a checkpoint list in {index_path}")

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".generate-levels-", dir=output) as temporary:
        staging = Path(temporary)
        for split, entries in (("train", fitting), ("validation", validation)):
            samples = select_preview_levels(entries, selection_seed)
            torch.manual_seed(posterior_seed)
            if device.type == "mps":
                torch.mps.manual_seed(posterior_seed)
            manifest = save_checkpoint_reconstructions(
                model, data, samples, staging / split, device, posterior_seed,
                checkpoint_relative, epoch, selection_seed=selection_seed,
                categorical_tiles=categorical, split=split,
            )
            record[split] = {
                "sample_count": manifest["sample_count"],
                "comparison_image": (f"{split}/{path.stem}/{manifest['comparison_image']}"
                                     if manifest["comparison_image"] is not None else None),
                "samples_file": f"{split}/{path.stem}/samples.json",
            }
        manifest = save_generated_levels(
            model, data.channels, staging / "generated", device, generation_seed,
            checkpoint_relative, epoch, latent_dim=metadata["latent_dim"],
            categorical_tiles=categorical,
        )
        record["generated"] = {"sample_count": manifest["sample_count"],
                               "comparison_image": f"generated/{path.stem}/comparison.png",
                               "samples_file": f"generated/{path.stem}/samples.json"}
        index = [item for item in index if item["checkpoint"] != checkpoint_relative]
        index.append(record)
        index.sort(key=lambda item: (item["epoch"], item["checkpoint"]))
        publish_outputs(staging, output, path.stem, index)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    search = parser.add_mutually_exclusive_group()
    search.add_argument("--runs-dir", type=Path, help="Recursively search this root; default model-training")
    search.add_argument("--checkpoints-dir", type=Path, help="Process the .pt files in one checkpoint folder")
    parser.add_argument("--output-dir", type=Path,
                        help="Output root; recursive searches preserve relative run folders; default beside checkpoints")
    parser.add_argument("--dataset", type=Path, help="Override the checkpoint's dataset location")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, help="Override the saved seed for input and latent sampling")
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32 - 1]")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    root = (args.checkpoints_dir or args.runs_dir or HERE).resolve()
    checkpoints = discover_checkpoints(root, checkpoints_only=args.checkpoints_dir is not None)
    if not checkpoints:
        raise ValueError(f"No saved .pt checkpoints found under {root}")
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    device = torch.device(requested)
    torch.set_num_threads(args.cpu_threads)
    failures = []
    print(f"Found {len(checkpoints)} checkpoints; device={device}", flush=True)
    for number, path in enumerate(checkpoints, 1):
        try:
            output = path.parent.parent / "reconstructions"
            if args.output_dir is not None:
                output = args.output_dir.resolve()
                if args.checkpoints_dir is None and root.name != "checkpoints":
                    output /= path.parent.parent.relative_to(root)
            record = process_checkpoint(path, device, args.dataset, args.seed, output_dir=output)
            counts = ", ".join(f"{split}={record[split]['sample_count']}" for split in SPLITS)
            print(f"[{number}/{len(checkpoints)}] {path}: {counts}", flush=True)
            for split in SPLITS:
                image = record[split]["comparison_image"]
                if image is not None:
                    print(f"  {output / image}", flush=True)
        except Exception as error:
            failures.append(path)
            print(f"[{number}/{len(checkpoints)}] Failed {path}: {error}", file=sys.stderr, flush=True)
    print(f"Completed {len(checkpoints) - len(failures)}/{len(checkpoints)} checkpoints", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
