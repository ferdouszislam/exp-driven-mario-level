"""Render VAE reconstructions and levels sampled from saved-model decoders."""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import torch

from categorical_reconstruction import reconstruction_probabilities
from data import INPUT_SHAPE


PREVIEW_COUNT = 5
RENDERER_PATH = (Path(__file__).resolve().parents[1]
                 / "datasets/fun-study-dataset-analysis/encoding_to_level_image.py")
SELECTION_POLICY = (
    "First playthrough per unique level_id in sorted {split} instances/playthroughs; "
    "Random(selection_seed).sample over sorted level IDs; same inputs for every checkpoint."
)


def select_preview_levels(entries: list[dict], seed: int,
                          count: int = PREVIEW_COUNT) -> list[dict]:
    """Pick distinct level IDs, retaining a path input from the supplied portion."""
    if type(count) is not int or count < 1:
        raise ValueError("Preview count must be a positive integer.")
    candidates = {}
    for entry in sorted(entries, key=lambda item: item["instance_id"]):
        for play in sorted(entry["playthroughs"],
                           key=lambda item: (item["level_id"], item["playthrough_key"])):
            candidates.setdefault(play["level_id"], {
                "level_id": play["level_id"],
                "playthrough_key": play["playthrough_key"],
                "instance_id": entry["instance_id"],
                "encoding": play["files"]["encoding"],
            })
    keys = random.Random(seed).sample(sorted(candidates), min(count, len(candidates)))
    return [candidates[key] for key in keys]


def select_validation_levels(validation: list[dict], seed: int,
                             count: int = PREVIEW_COUNT) -> list[dict]:
    """Keep the validation-only selector interface for existing callers."""
    return select_preview_levels(validation, seed, count)


@lru_cache(maxsize=1)
def _renderer():
    """Load the project's existing renderer without modifying Python's paths."""
    spec = importlib.util.spec_from_file_location("mario_encoding_renderer", RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load level renderer: {RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encoding_to_image


def _decoding_policy(categorical_tiles: bool) -> str:
    if categorical_tiles:
        return "argmax over 21 tile softmax probabilities; independent sigmoid path probability >= 0.5"
    return "argmax over 21 tile logits; independent path bit at logit >= 0 (probability >= 0.5)"


def _binary_encodings(logits: torch.Tensor, categorical_tiles: bool) -> np.ndarray:
    """Convert decoder logits to one geometry tile and an independent path bit."""
    if categorical_tiles:
        probabilities = reconstruction_probabilities(logits)
        tile_ids = probabilities[:, :21].argmax(dim=1).cpu().numpy()
        path_bits = (probabilities[:, 21] >= 0.5).to(torch.uint8).cpu().numpy()
    else:
        tile_ids = logits[:, :21].argmax(dim=1).cpu().numpy()
        path_bits = (logits[:, 21] >= 0).to(torch.uint8).cpu().numpy()
    geometry = np.eye(21, dtype=np.uint8)[tile_ids].transpose(0, 3, 1, 2)
    return np.concatenate((geometry, path_bits[:, None]), axis=1)


def save_checkpoint_reconstructions(model, data, samples: list[dict], output: Path,
                                    device: torch.device, seed: int,
                                    checkpoint: str, epoch: int, *, selection_seed: int | None = None,
                                    categorical_tiles: bool = False,
                                    split: str = "validation") -> dict:
    """Save paired level PNGs from one posterior sample for each supplied input.

    The caller seeds posterior sampling. This function restores the model's
    previous training/evaluation mode.
    """
    if split not in ("train", "validation"):
        raise ValueError("Preview split must be 'train' or 'validation'.")
    input_label = "Training" if split == "train" else "Validation"
    record = {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "epoch": epoch,
        "split": split,
        "seed": seed,
        "selection_seed": selection_seed,
        "requested_sample_count": PREVIEW_COUNT,
        "sample_count": len(samples),
        "selection_policy": SELECTION_POLICY.format(split=split),
        "latent_policy": f"one reparameterized posterior sample per {split} input",
        "decoding": _decoding_policy(categorical_tiles),
        "channels_metadata": "channels.json" if samples else None,
        "comparison_image": "comparison.png" if samples else None,
        "samples": [],
    }
    if not samples:
        record["skipped_reason"] = f"No {split} levels are available."
        output = Path(output)
        output.mkdir(parents=True)
        (output / "samples.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record

    from PIL import Image, ImageDraw, ImageFont

    render = _renderer()
    inputs = torch.stack([data.tensors[sample["encoding"]] for sample in samples]).to(device)
    previous_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            mu, log_var = model.encode(inputs)
            logits = model.decode(model.reparameterize(mu, log_var))
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite VAE reconstruction logits in checkpoint preview.")
        encodings = _binary_encodings(logits, categorical_tiles)
    finally:
        model.train(previous_training)

    output = Path(output)
    output.mkdir(parents=True)
    channels_path = output / "channels.json"
    channels_path.write_text(json.dumps(data.channels, indent=2) + "\n", encoding="utf-8")
    pairs = []
    for index, sample in enumerate(samples, 1):
        generated = encodings[index - 1]
        original = inputs[index - 1].detach().cpu().numpy().astype(np.uint8)
        input_image = render(original, channels_path=channels_path)
        generated_image = render(generated, channels_path=channels_path)
        filenames = {
            "input_image": f"{index:02d}_input.png",
            "generated_image": f"{index:02d}_generated.png",
            "generated_encoding": f"{index:02d}_generated.npy",
        }
        input_image.save(output / filenames["input_image"], format="PNG")
        generated_image.save(output / filenames["generated_image"], format="PNG")
        np.save(output / filenames["generated_encoding"], generated, allow_pickle=False)
        record["samples"].append({**sample, "sample_number": index, **filenames})
        pairs.append((sample, input_image, generated_image))

    label_height = 36
    width, height = pairs[0][1].size
    row_height = 2 * (height + label_height)
    sheet = Image.new("RGB", (width, row_height * len(pairs)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=24)
    for index, (sample, original, generated) in enumerate(pairs):
        top = index * row_height
        draw.text((8, top + 4), f"{index + 1}. {input_label} input: {sample['level_id']}", fill="black", font=font)
        sheet.paste(original, (0, top + label_height))
        draw.text((8, top + height + label_height + 4),
                  f"VAE reconstruction | epoch {epoch}", fill="black", font=font)
        sheet.paste(generated, (0, top + height + 2 * label_height))
    sheet.save(output / record["comparison_image"], format="PNG")
    (output / "samples.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def save_generated_levels(model, channels: dict, output: Path, device: torch.device,
                          seed: int, checkpoint: str, epoch: int, *, latent_dim: int,
                          categorical_tiles: bool = False, count: int = PREVIEW_COUNT) -> dict:
    """Decode independent standard-normal latent vectors without encoding inputs."""
    if type(count) is not int or count < 1:
        raise ValueError("Generated level count must be a positive integer.")
    if type(latent_dim) is not int or latent_dim < 1:
        raise ValueError("Latent dimension must be a positive integer.")

    from PIL import Image, ImageDraw, ImageFont

    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(count, latent_dim, generator=generator, dtype=torch.float32)
    previous_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            logits = model.decode(latent.to(device))
        if logits.shape != (count, *INPUT_SHAPE):
            raise ValueError(f"Expected generated logits with shape {(count, *INPUT_SHAPE)}, got {tuple(logits.shape)}.")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite VAE decoder logits in generated levels.")
        encodings = _binary_encodings(logits, categorical_tiles)
    finally:
        model.train(previous_training)

    output = Path(output)
    output.mkdir(parents=True)
    channels_path = output / "channels.json"
    channels_path.write_text(json.dumps(channels, indent=2) + "\n", encoding="utf-8")
    record = {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "epoch": epoch,
        "split": "generated",
        "seed": seed,
        "requested_sample_count": count,
        "sample_count": count,
        "latent_dim": latent_dim,
        "latent_policy": "z ~ N(0, I); decoder-only generation without an input level",
        "decoding": _decoding_policy(categorical_tiles),
        "channels_metadata": "channels.json",
        "comparison_image": "comparison.png",
        "samples": [],
    }
    render = _renderer()
    images = []
    for index, encoding in enumerate(encodings, 1):
        filenames = {
            "generated_image": f"{index:02d}_generated.png",
            "generated_encoding": f"{index:02d}_generated.npy",
            "latent_vector": f"{index:02d}_latent.npy",
        }
        generated_image = render(encoding, channels_path=channels_path)
        generated_image.save(output / filenames["generated_image"], format="PNG")
        np.save(output / filenames["generated_encoding"], encoding, allow_pickle=False)
        np.save(output / filenames["latent_vector"], latent[index - 1].numpy(), allow_pickle=False)
        record["samples"].append({"sample_number": index, **filenames})
        images.append(generated_image)

    label_height = 36
    width, height = images[0].size
    row_height = height + label_height
    sheet = Image.new("RGB", (width, row_height * count), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=24)
    for index, generated_image in enumerate(images):
        top = index * row_height
        draw.text((8, top + 4), f"{index + 1}. Generated level | epoch {epoch}", fill="black", font=font)
        sheet.paste(generated_image, (0, top + label_height))
    sheet.save(output / record["comparison_image"], format="PNG")
    (output / "samples.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
