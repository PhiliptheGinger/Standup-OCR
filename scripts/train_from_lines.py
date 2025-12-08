"""Batch training helper for line-level handwriting datasets.

This script fine-tunes a Tesseract model using a folder of line images
paired with `.gt.txt` ground-truth files (Kraken/PAGE-XML style exports).

Compared to `main.py train` (which derives labels from file names or GPT),
this path reads the transcription from the existing `<image>.gt.txt` file
next to each PNG and skips any external API usage.

Workflow:
1. (Optional) Normalize orientation so all line images are landscape.
2. Generate / regenerate `.lstmf` files for each line using `_generate_lstmf`.
3. Validate the `.lstmf` files.
4. Run `combine_tessdata` + `lstmtraining` continuation or scratch.
5. Produce `<output_model>.traineddata` inside the model directory.

Usage (PowerShell):
    python scripts/train_from_lines.py \
        --lines-dir train/lines \
        --output-model handwriting_lines \
        --model-dir models \
        --base-lang eng \
        --max-iterations 8000 \
        --normalize-orientation

Notes:
* Requires Tesseract training utilities (`lstmtraining`, `combine_tessdata`).
* Ensure `TESSDATA_PREFIX` or `--tessdata-dir` points to language packs.
* Orientation normalization rewrites the PNGs in-place (make a backup first if desired).
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import List, Optional

# Ensure repo root import resolution
import sys
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.training import (
    _generate_lstmf,
    _validate_lstmf,
    _resolve_tessdata_dir,
    _is_fast_model,
    _get_unicharset_size,
    _ensure_directory,
)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def discover_line_images(folder: Path) -> List[Path]:
    images: List[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            if p.with_suffix(".gt.txt").exists():
                images.append(p)
            else:
                logging.warning("Skipping %s (missing .gt.txt)", p.name)
    return images


def normalize_orientation(image_paths: List[Path], *, threshold: float = 1.1) -> None:
    from PIL import Image, ImageOps
    rotated = 0
    for p in image_paths:
        try:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img)  # respect EXIF if present
            w, h = img.size
            if h > w * threshold:
                img = img.rotate(90, expand=True)
                img.save(p)
                rotated += 1
                logging.info("Rotated %s -> %s", p.name, img.size)
        except Exception as exc:
            logging.debug("Orientation skip for %s: %s", p.name, exc)
    logging.info("Orientation normalization complete (%d rotated)", rotated)


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a handwriting model from line images + .gt.txt files")
    p.add_argument("--lines-dir", type=Path, default=Path("train/lines"), help="Directory containing line images + .gt.txt files")
    p.add_argument("--output-model", required=True, help="Base name for output .traineddata (e.g. handwriting_lines)")
    p.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory to store training artifacts")
    p.add_argument("--tessdata-dir", type=Path, help="Path to tessdata (overrides auto discovery)")
    p.add_argument("--base-lang", default="eng", help="Base language (default: eng)")
    p.add_argument("--max-iterations", type=int, default=5000, help="Maximum training iterations (default: 5000)")
    p.add_argument("--normalize-orientation", action="store_true", help="Rotate portrait images counterclockwise to landscape before training")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    return p


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    logging.info("Running: %s", " ".join(str(x) for x in command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        logging.debug(result.stdout.strip())
    if result.stderr:
        logging.debug(result.stderr.strip())
    return result


def ensure_success(result: subprocess.CompletedProcess[str], description: str) -> None:
    if result.returncode == 0:
        return
    output = ((result.stderr or "") + (result.stdout or "")).strip()
    raise RuntimeError(f"{description} failed: {output or f'code {result.returncode}'}")


def main() -> None:
    args = build_argument_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    lines_dir: Path = args.lines_dir
    if not lines_dir.exists():
        raise FileNotFoundError(f"Lines directory not found: {lines_dir}")

    images = discover_line_images(lines_dir)
    if not images:
        raise RuntimeError(f"No training images with .gt.txt found in {lines_dir}")
    logging.info("Discovered %d line image(s)", len(images))

    if args.normalize_orientation:
        logging.info("Normalizing orientation...")
        normalize_orientation(images)

    model_dir: Path = args.model_dir
    work_dir = _ensure_directory(model_dir / f"{args.output_model}_training")
    tessdata_dir = _resolve_tessdata_dir(args.tessdata_dir)

    lstmf_paths: List[Path] = []
    for img in images:
        try:
            lstmf = _generate_lstmf(img, work_dir, args.base_lang)
        except Exception as exc:
            logging.exception("Failed to generate lstmf for %s", img.name)
            continue
        if lstmf:
            lstmf_paths.append(lstmf)

    if not lstmf_paths:
        raise RuntimeError("No .lstmf files generated; aborting")

    valid: List[Path] = []
    invalid: List[Path] = []
    for p in lstmf_paths:
        if _validate_lstmf(p):
            valid.append(p)
        else:
            invalid.append(p)
    if invalid:
        logging.warning("Excluded %d invalid .lstmf file(s)", len(invalid))
    if not valid:
        raise RuntimeError("All .lstmf files failed validation")

    list_file = work_dir / "training_files.txt"
    list_file.write_text("\n".join(str(p) for p in valid) + "\n", encoding="utf-8")

    base_traineddata = tessdata_dir / f"{args.base_lang}.traineddata"
    if not base_traineddata.exists():
        raise FileNotFoundError(f"Missing base traineddata: {base_traineddata}")

    extracted_dir = _ensure_directory(work_dir / "extracted")
    combine_prefix = extracted_dir / args.base_lang
    ensure_success(run_command(["combine_tessdata", "-u", str(base_traineddata), str(combine_prefix)]), "combine_tessdata")

    lstm_path = combine_prefix.with_suffix(".lstm")
    if not lstm_path.exists():
        raise RuntimeError("combine_tessdata did not produce .lstm")

    checkpoint_prefix = work_dir / f"{args.output_model}_checkpoint"
    fast_model = _is_fast_model(args.base_lang, base_traineddata, extracted_dir)
    unicharset_size = _get_unicharset_size(base_traineddata, extracted_dir, args.base_lang)

    if unicharset_size is None:
        raise RuntimeError("Unable to determine unicharset size from base traineddata")
    if unicharset_size == 113:  # legacy mismatch workaround
        logging.warning("Overriding unicharset_size 113 -> 111 for legacy traineddata")
        unicharset_size = 111

    net_spec = f"[1,48,0,1 Lfx128 O1c{unicharset_size}]"

    continue_cmd = [
        "lstmtraining",
        "--continue_from", str(lstm_path),
        "--model_output", str(checkpoint_prefix),
        "--traineddata", str(base_traineddata),
        "--train_listfile", str(list_file),
        "--max_iterations", str(args.max_iterations),
    ]
    scratch_cmd = [
        "lstmtraining",
        "--net_spec", net_spec,
        "--model_output", str(checkpoint_prefix),
        "--traineddata", str(base_traineddata),
        "--train_listfile", str(list_file),
        "--max_iterations", str(args.max_iterations),
    ]

    if fast_model:
        logging.info("Base model is integer-only; training from scratch")
        ensure_success(run_command(scratch_cmd), "lstmtraining scratch")
    else:
        result = run_command(continue_cmd)
        if result.returncode != 0:
            logging.warning("Falling back to scratch training due to continuation failure")
            ensure_success(run_command(scratch_cmd), "lstmtraining scratch")
        else:
            logging.info("Continuation training complete")

    checkpoint_file = checkpoint_prefix.with_suffix(".checkpoint")
    if not checkpoint_file.exists():
        raise RuntimeError("Checkpoint file missing after training")

    final_model = work_dir / f"{args.output_model}.traineddata"
    ensure_success(run_command([
        "lstmtraining",
        "--stop_training",
        "--continue_from", str(checkpoint_file),
        "--traineddata", str(base_traineddata),
        "--model_output", str(final_model),
    ]), "stop_training")

    target_model = model_dir / f"{args.output_model}.traineddata"
    target_model.write_bytes(final_model.read_bytes())
    logging.info("Training complete: %s", target_model)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
