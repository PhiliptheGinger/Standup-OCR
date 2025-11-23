#!/usr/bin/env python3
"""Fine-tune the Standup-OCR model on a structured JSON dataset."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple

from src.training import train_model

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_pages(json_path: Path) -> list[dict]:
    """Load and normalize page entries from the dataset JSON file.

    Supports both a bare list of entries and the structured format that
    wraps entries inside a top-level object containing metadata such as
    notebooks/file_numbers. Non-dictionary entries are skipped to avoid
    runtime errors when preparing training pairs.
    """

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        raw_entries = data
    elif isinstance(data, dict) and isinstance(data.get("entries"), list):
        raw_entries = data["entries"]
    else:
        raise ValueError("JSON format not recognized: expected a list or an object with an 'entries' list")

    entries: list[dict] = []
    for idx, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            logging.warning("Skipping non-dictionary entry at index %d in JSON", idx)
            continue
        entries.append(entry)
    return entries


def build_training_text(entry: dict) -> str:
    """Construct the training string from a single page entry."""

    indent_levels = {
        "main": 0,
        "sub": 1,
        "subsub": 2,
        "sub3": 3,
    }

    segments: List[str] = []
    title = entry.get("title")
    if title:
        segments.append(str(title))
    for line in entry.get("lines", []):
        arrow = line.get("arrow", "main")
        text = line.get("text", "")
        if not text:
            continue
        level = indent_levels.get(arrow, 1 if arrow != "main" else 0)
        prefix = "\t" * level
        segments.append(f"{prefix}{text}")
    return "\n".join(segments)


def _numeric_candidates(number: int | None, fallback_index: int) -> Iterable[str]:
    """Yield common numeric filename variants for a page or file number."""

    if number is None:
        number = fallback_index + 1

    yield f"{number}"
    yield f"{number:03d}"
    yield f"page{number}"
    yield f"page{number:03d}"


def locate_image(entry: dict, images_dir: Path, index: int, offset: int) -> Path:
    """Find the matching image file for a dataset entry."""

    images_dir = images_dir.resolve()
    if not isinstance(entry, dict):
        raise TypeError(f"Entry at index {index} is not a dictionary; received {type(entry).__name__}")

    page_value = entry.get("page")

    file_number = entry.get("file_number")
    if isinstance(file_number, int):
        actual_number = file_number + offset
    else:
        actual_number = None

    candidate_names: list[str] = []
    if actual_number is not None:
        candidate_names.extend(_numeric_candidates(actual_number, index))
    candidate_names.extend(_numeric_candidates(page_value if isinstance(page_value, int) else None, index))

    for name in candidate_names:
        for ext in SUPPORTED_EXTENSIONS:
            candidate = images_dir / f"{name}{ext}"
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Could not find image for entry {index + 1} (page={page_value}, file_number={file_number}) in {images_dir}"
    )


def prepare_training_pairs(entries: list, images_dir: Path, train_dir: Path, offset: int) -> list[Path]:
    """Write .gt.txt files and copy images to the training directory."""

    train_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            logging.warning("Skipping non-dictionary entry at index %d", idx)
            continue

        logging.info("Processing entry %d", idx + 1)
        image_path = locate_image(entry, images_dir, idx, offset)
        text = build_training_text(entry)
        if not text.strip():
            logging.warning("Skipping empty transcription for entry %d (%s)", idx + 1, image_path.name)
            continue
        destination = train_dir / image_path.name
        shutil.copy2(image_path, destination)
        gt_path = destination.with_suffix(".gt.txt")
        with gt_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text.strip() + "\n")
        logging.info("Prepared training pair for %s", destination.name)
        written.append(destination)
    return written


def split_dataset(paths: list[Path], validation_split: float, seed: int) -> tuple[list[Path], list[Path]]:
    """Split the dataset into train and validation subsets."""
    if not 0.0 < validation_split < 1.0:
        return paths, []
    random.seed(seed)
    random.shuffle(paths)
    cutoff = int(len(paths) * (1.0 - validation_split))
    return paths[:cutoff], paths[cutoff:]


def relocate_validation_pairs(validation_pairs: list[Path], train_dir: Path) -> list[Path]:
    """Move validation samples to a dedicated folder to keep training data clean."""
    if not validation_pairs:
        return []
    val_dir = train_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for image_path in validation_pairs:
        destination = val_dir / image_path.name
        shutil.move(image_path, destination)
        gt_src = image_path.with_suffix(".gt.txt")
        gt_dest = destination.with_suffix(".gt.txt")
        if gt_src.exists():
            shutil.move(gt_src, gt_dest)
        moved.append(destination)
    return moved


def evaluate_model(model_name: str, model_dir: Path, validation_pairs: list[Path]) -> float:
    """Run Tesseract on validation images and return average character accuracy."""
    if not validation_pairs:
        return 0.0
    correct = 0
    total = 0
    for image_path in validation_pairs:
        gt_path = image_path.with_suffix(".gt.txt")
        if not gt_path.exists():
            logging.warning("Missing GT file for validation image %s", image_path.name)
            continue
        ground_truth = gt_path.read_text(encoding="utf-8").replace("\r", "").strip()
        try:
            predicted = run_tesseract_ocr(image_path, model_name, model_dir)
        except RuntimeError as exc:
            logging.warning("Tesseract evaluation failed for %s: %s", image_path.name, exc)
            continue
        if not ground_truth:
            continue
        correct += count_matching_characters(ground_truth, predicted)
        total += len(ground_truth)
    if total == 0:
        return 0.0
    return correct / total


def run_tesseract_ocr(image_path: Path, model_name: str, model_dir: Path) -> str:
    """Run Tesseract with the fine-tuned model and return the OCR text."""
    command = [
        "tesseract",
        str(image_path),
        "stdout",
        "-l",
        model_name,
        "--tessdata-dir",
        str(model_dir),
        "--psm",
        "6",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Tesseract is not installed or not in PATH") from exc
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip())
    return (result.stdout or "").replace("\r", "").strip()


def count_matching_characters(reference: str, candidate: str) -> int:
    """Count character-level matches between two strings."""
    return sum(1 for ref_char, cand_char in zip(reference, candidate) if ref_char == cand_char)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the Standup-OCR model on a structured dataset")
    parser.add_argument("--json", required=True, type=Path, help="Path to standup_option_c.json")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing page images")
    parser.add_argument("--train-dir", default=Path("data/train/images"), type=Path, help="Output directory for training pairs")
    parser.add_argument("--offset", type=int, default=0, help="Offset to add to file_number when deriving image filenames")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory (and model name) for training artefacts, e.g. models/finetuned")
    parser.add_argument("--base-model", default="eng.traineddata", help="Base traineddata file or language code")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum training iterations")
    parser.add_argument("--validation-split", type=float, default=0.0, help="Optional validation split (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def derive_model_paths(output_dir: Path) -> tuple[str, Path]:
    """Return the output model name and parent directory."""
    output_dir = output_dir.resolve()
    if output_dir.suffix:
        return output_dir.stem, output_dir.parent
    return output_dir.name, output_dir.parent


def resolve_base_lang_and_dir(base_model_arg: str) -> tuple[str, Path | None]:
    """Derive the base language code and tessdata directory from CLI input."""
    base_path = Path(base_model_arg)
    if base_path.suffix:
        return base_path.stem, base_path.parent if base_path.parent != Path("") else None
    return base_model_arg, None


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    logging.info("Loading dataset from %s", args.json)
    entries = load_pages(args.json)
    logging.info("Loaded %d page entries", len(entries))

    train_dir = args.train_dir
    prepared_images = prepare_training_pairs(entries, args.images, train_dir, args.offset)

    train_subset, val_subset = split_dataset(prepared_images, args.validation_split, args.seed)
    if val_subset:
        val_subset = relocate_validation_pairs(val_subset, train_dir)
        logging.info("Reserved %d sample(s) for validation", len(val_subset))
    if train_subset != prepared_images:
        logging.info("Using %d sample(s) for training", len(train_subset))

    output_model, model_dir = derive_model_paths(args.output_dir)
    base_lang, tessdata_dir = resolve_base_lang_and_dir(args.base_model)

    train_result = train_model(
        train_dir=train_dir,
        output_model=output_model,
        model_dir=model_dir if str(model_dir) else None,
        base_lang=base_lang,
        max_iterations=args.max_iter,
        use_gpt_ocr=False,
        tessdata_dir=tessdata_dir,
    )

    logging.info("Training finished: %s", train_result)

    if val_subset:
        accuracy = evaluate_model(output_model, model_dir, val_subset)
        logging.info("Validation character accuracy: %.2f%%", accuracy * 100)

    logging.info("Processed %d training pair(s) from %s", len(prepared_images), args.json)


if __name__ == "__main__":
    main()
