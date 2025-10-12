"""Standalone CLI helpers for the agentic OCR workflow."""
from __future__ import annotations

import argparse
import csv
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from . import ocr_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
READY_FOR_AGENT_DIR = PROJECT_ROOT / "uploads" / "ready_for_agent"
TRANSCRIPTS_RAW_DIR = PROJECT_ROOT / "transcripts" / "raw"
ANNOTATION_LOG = PROJECT_ROOT / "train" / "annotation_log.csv"
READY_ZIP = PROJECT_ROOT / "ready_for_review.zip"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

DEFAULT_MODEL_NAME = "microsoft/trocr-base-handwritten"


processor = TrOCRProcessor.from_pretrained(DEFAULT_MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(DEFAULT_MODEL_NAME)
model.eval()


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Return cached TrOCR processor/model instances for ``model_name``."""

    if model_name == DEFAULT_MODEL_NAME:
        return processor, model
    loaded_processor = TrOCRProcessor.from_pretrained(model_name)
    loaded_model = VisionEncoderDecoderModel.from_pretrained(model_name)
    loaded_model.eval()
    return loaded_processor, loaded_model


def transcribe_image(img_path: Path, model_name: str = DEFAULT_MODEL_NAME) -> str:
    image = Image.open(img_path).convert("RGB")
    trocr_processor, trocr_model = _load_model(model_name)
    pixel_values = trocr_processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = trocr_model.generate(pixel_values)
    transcription = trocr_processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return transcription.strip()


@dataclass
class AgenticBatchResult:
    """Container describing the outcome of :func:`agentic_batch_transcribe`."""

    processed: int
    skipped: int
    zip_path: Path


def _iter_ready_images() -> Iterable[Path]:
    if not READY_FOR_AGENT_DIR.exists():
        return []
    images = [
        path
        for path in sorted(READY_FOR_AGENT_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return images


def _transcribe_image(image_path: Path, *, model_name: str) -> str:
    try:
        logging.debug("Running TrOCR (%s) for %s", model_name, image_path.name)
        return transcribe_image(image_path, model_name=model_name)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logging.exception("TrOCR failed for %s: %s", image_path.name, exc)
        logging.info("Falling back to pytesseract for %s", image_path.name)
        return ocr_image(image_path)


def _append_annotation_log(page: str, transcription: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    ANNOTATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    file_exists = ANNOTATION_LOG.exists()
    with ANNOTATION_LOG.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(["page", "transcription", "timestamp"])
        writer.writerow([page, transcription, timestamp])


def _zip_raw_transcripts() -> Path:
    TRANSCRIPTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(TRANSCRIPTS_RAW_DIR.glob("*.txt"))
    with zipfile.ZipFile(READY_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            arcname = file_path.relative_to(PROJECT_ROOT)
            zf.write(file_path, arcname)
    return READY_ZIP


def agentic_batch_transcribe(
    *, rerun_failed: bool = False, force: bool = False, model_name: str = DEFAULT_MODEL_NAME
) -> AgenticBatchResult:
    """Process scans in :mod:`uploads/ready_for_agent` and prepare transcripts."""

    TRANSCRIPTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    for image_path in _iter_ready_images():
        transcript_path = TRANSCRIPTS_RAW_DIR / f"{image_path.stem}.txt"
        if transcript_path.exists() and not force:
            if rerun_failed and not transcript_path.read_text(encoding="utf-8").strip():
                logging.info("Reprocessing empty transcript for %s", image_path.name)
            else:
                logging.debug("Skipping %s (transcript exists)", image_path.name)
                skipped += 1
                continue

        text = _transcribe_image(image_path, model_name=model_name)
        transcript_path.write_text(text, encoding="utf-8")
        _append_annotation_log(image_path.name, text)
        processed += 1

    zip_path = _zip_raw_transcripts()
    logging.info(
        "[TrOCR] Processed %d file(s), skipped %d. Archive: %s. Log: %s",
        processed,
        skipped,
        zip_path.name,
        ANNOTATION_LOG.relative_to(PROJECT_ROOT),
    )
    return AgenticBatchResult(processed=processed, skipped=skipped, zip_path=zip_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic OCR helpers")
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Run the agentic batch transcription pipeline",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Re-run scans whose transcripts exist but are empty",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all scans, even if transcripts already exist",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Hugging Face model identifier for TrOCR (default:"
            f" {DEFAULT_MODEL_NAME})."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not any([args.agentic, args.rerun_failed, args.force]):
        parser.print_help()
        return

    if args.agentic and args.rerun_failed and not args.force:
        logging.warning(
            "--agentic already processes new files; --rerun-failed will only re-run empty transcripts."
        )

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    result = agentic_batch_transcribe(
        rerun_failed=args.rerun_failed, force=args.force, model_name=args.model
    )
    logging.info(
        "Processed %d file(s), skipped %d. Archive available at %s",
        result.processed,
        result.skipped,
        result.zip_path,
    )


if __name__ == "__main__":
    main()
