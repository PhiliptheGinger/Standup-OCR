"""Utilities for preparing Kraken line datasets."""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import json

from .preprocessing import preprocess_image
from kraken import binarization, pageseg
from kraken.lib import xml

log = logging.getLogger(__name__)

_LINE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class LineSanitizationStats:
    """Summary of the sanitize_line_dataset run."""

    processed: int = 0
    missing_gt: int = 0
    empty_gt: int = 0


def _iter_line_images(source_dir: Path) -> Iterable[Path]:
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _LINE_EXTENSIONS:
            yield path


def _resolve_transcription(image_path: Path) -> Path | None:
    gt_path = image_path.with_suffix(".gt.txt")
    if gt_path.exists():
        return gt_path
    txt_path = image_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path
    return None


def sanitize_line_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    adaptive: bool = True,
    force_landscape: bool = True,
    resize_width: int | None = None,
) -> LineSanitizationStats:
    """Rebuild Kraken line crops using the preprocessing pipeline.

    Parameters
    ----------
    source_dir:
        Directory containing the existing Kraken ``lines`` dataset.
    output_dir:
        Where the cleaned images plus ``.gt.txt`` files should be written.
    adaptive:
        Whether to use adaptive thresholding (default: True).
    force_landscape:
        Rotate portrait lines counter-clockwise so Kraken sees upright text.
    resize_width:
        Optional width used by :func:`preprocess_image`. When ``None`` or ``0``
        the function keeps the original resolution of each line image.
    """

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        output_root = output_dir.resolve()
    except OSError:
        output_root = output_dir

    stats = LineSanitizationStats()
    resize_value = resize_width or 0

    for image_path in _iter_line_images(source_dir):
        try:
            if image_path.resolve().is_relative_to(output_root):
                continue
        except (OSError, AttributeError):
            pass
        label_path = _resolve_transcription(image_path)
        if label_path is None:
            log.warning("Skipping %s (no matching transcription file)", image_path)
            stats.missing_gt += 1
            continue

        text = label_path.read_text(encoding="utf8")
        if not text.strip():
            log.warning("Skipping %s (empty transcription)", image_path)
            stats.empty_gt += 1
            continue

        processed = preprocess_image(
            image_path,
            resize_width=resize_value,
            adaptive=adaptive,
            force_landscape=force_landscape,
        )
        out_image = output_dir / image_path.name
        Image.fromarray(processed).save(out_image)

        out_label = output_dir / f"{image_path.stem}.gt.txt"
        out_label.write_text(text, encoding="utf8")
        stats.processed += 1

        metadata_source = image_path.with_suffix(".boxes.json")
        if metadata_source.exists():
            metadata_target = output_dir / metadata_source.name
            if not metadata_target.exists():
                shutil.copy2(metadata_source, metadata_target)

    if stats.processed == 0:
        log.warning(
            "No valid line crops were produced from %s. Check that the directory "
            "contains .png/.jpg images with accompanying .gt.txt files.",
            source_dir,
        )
    else:
        log.info(
            "Cleaned %d line images (skipped %d missing GT, %d empty GT)",
            stats.processed,
            stats.missing_gt,
            stats.empty_gt,
        )

    return stats


def _get_segmentation(
    image_path: Path,
    *,
    model: str | None = None,
    out_pagexml: Path | None = None,
) -> dict[str, Any]:
    """Always invoke `kraken segment` CLI and parse its JSON output."""
    segmentation = _segment_via_cli(image_path, model=model)

    if out_pagexml:
        out_pagexml.parent.mkdir(parents=True, exist_ok=True)
        out_pagexml.write_text(
            json.dumps(segmentation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return segmentation
