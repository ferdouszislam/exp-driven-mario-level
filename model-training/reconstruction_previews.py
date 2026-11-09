"""Render held-out inputs and their VAE reconstructions beside each checkpoint."""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import torch


PREVIEW_COUNT = 5
RENDERER_PATH = (Path(__file__).resolve().parents[1]
                 / "datasets/fun-study-dataset-analysis/encoding_to_level_image.py")
SELECTION_POLICY = (
    "First playthrough per unique level_id in sorted validation instances/playthroughs; "
    "Random(selection_seed).sample over sorted level IDs; same inputs for every checkpoint."
)


def select_validation_levels(validation: list[dict], seed: int,
                             count: int = PREVIEW_COUNT) -> list[dict]:
    """Pick distinct level IDs, retaining the associated validation path input."""
    if type(count) is not int or count < 1:
        raise ValueError("Preview count must be a positive integer.")
    candidates = {}
    for entry in sorted(validation, key=lambda item: item["instance_id"]):
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


@lru_cache(maxsize=1)
def _renderer():
    """Load the project's existing renderer without modifying Python's paths."""
    spec = importlib.util.spec_from_file_location("mario_encoding_renderer", RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load level renderer: {RENDERER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encoding_to_image


def save_checkpoint_reconstructions(model, data, samples: list[dict], output: Path,
                                    device: torch.device, seed: int,
                                    checkpoint: str, epoch: int, *, selection_seed: int | None = None) -> dict:
    """Save paired level PNGs from one posterior sample for each validation input.

    The caller seeds and preserves the training RNG streams around this reporting
    pass. This function restores the model's previous training/evaluation mode.
    """
    record = {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "epoch": epoch,
        "seed": seed,
        "selection_seed": selection_seed,
        "requested_sample_count": PREVIEW_COUNT,
        "sample_count": len(samples),
        "selection_policy": SELECTION_POLICY,
        "latent_policy": "one reparameterized posterior sample per validation input",
        "decoding": "argmax over 21 tile logits; independent path bit at logit >= 0 (probability >= 0.5)",
        "channels_metadata": "channels.json" if samples else None,
        "comparison_image": "comparison.png" if samples else None,
        "samples": [],
    }
    if not samples:
        record["skipped_reason"] = "No validation levels are available."
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
        tile_ids = logits[:, :21].argmax(dim=1).cpu().numpy()
        path_bits = (logits[:, 21] >= 0).to(torch.uint8).cpu().numpy()
    finally:
        model.train(previous_training)

    output = Path(output)
    output.mkdir(parents=True)
    channels_path = output / "channels.json"
    channels_path.write_text(json.dumps(data.channels, indent=2) + "\n", encoding="utf-8")
    pairs = []
    for index, sample in enumerate(samples, 1):
        generated = np.concatenate((
            np.eye(21, dtype=np.uint8)[tile_ids[index - 1]].transpose(2, 0, 1),
            path_bits[index - 1][None],
        ), axis=0)
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
        draw.text((8, top + 4), f"{index + 1}. Validation input: {sample['level_id']}", fill="black", font=font)
        sheet.paste(original, (0, top + label_height))
        draw.text((8, top + height + label_height + 4),
                  f"VAE reconstruction | epoch {epoch}", fill="black", font=font)
        sheet.paste(generated, (0, top + height + 2 * label_height))
    sheet.save(output / record["comparison_image"], format="PNG")
    (output / "samples.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
