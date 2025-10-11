"""Standalone CLI helpers for the agentic OCR workflow."""
from __future__ import annotations

import argparse
import logging
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ..kraken_adapter import is_available as kraken_available, ocr as kraken_ocr
from . import ocr_image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
READY_FOR_AGENT_DIR = PROJECT_ROOT / "uploads" / "ready_for_agent"
TRANSCRIPTS_RAW_DIR = PROJECT_ROOT / "transcripts" / "raw"
ANNOTATION_LOG = PROJECT_ROOT / "annotation.log"
READY_ZIP = PROJECT_ROOT / "ready_for_review.zip"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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


def _default_kraken_model() -> Path | None:
    if not kraken_available():
        return None
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        return None
    candidates = sorted(models_dir.glob("*.mlmodel"))
    if not candidates:
        return None
    return candidates[0]


def _transcribe_with_kraken(image_path: Path, model_path: Path) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        kraken_ocr(image_path, model_path, tmp_path)
        return tmp_path.read_text(encoding="utf-8").strip()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _transcribe_image(image_path: Path) -> str:
    model_path = _default_kraken_model()
    if model_path is not None:
        logging.info("Using Kraken model %s for %s", model_path.name, image_path.name)
        try:
            return _transcribe_with_kraken(image_path, model_path)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logging.exception("Kraken OCR failed for %s: %s", image_path.name, exc)
    logging.info("Falling back to pytesseract for %s", image_path.name)
    return ocr_image(image_path)


def _parse_notebook_and_page(stem: str) -> tuple[str, str]:
    match = re.match(r"(?P<notebook>.+?)_(?P<page>\d+)$", stem)
    if match:
        notebook = match.group("notebook")
        page = str(int(match.group("page")))
    else:
        notebook = stem
        page = "?"
    return notebook, page


def _append_annotation_log(notebook: str, page: str, image_name: str, transcript_path: Path) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    relative_transcript = transcript_path.relative_to(PROJECT_ROOT)
    line = f"[{timestamp}] {notebook} {page} {image_name} {relative_transcript.as_posix()}\n"
    with ANNOTATION_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _zip_raw_transcripts() -> Path:
    TRANSCRIPTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(TRANSCRIPTS_RAW_DIR.glob("*.txt"))
    with zipfile.ZipFile(READY_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            arcname = file_path.relative_to(PROJECT_ROOT)
            zf.write(file_path, arcname)
    return READY_ZIP


def agentic_batch_transcribe(*, rerun_failed: bool = False, force: bool = False) -> AgenticBatchResult:
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

        text = _transcribe_image(image_path)
        transcript_path.write_text(text, encoding="utf-8")
        notebook, page = _parse_notebook_and_page(image_path.stem)
        _append_annotation_log(notebook, page, image_path.name, transcript_path)
        processed += 1

    zip_path = _zip_raw_transcripts()
    logging.info(
        "Packaged %d transcript(s) (%d skipped) into %s", processed, skipped, zip_path.name
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

    result = agentic_batch_transcribe(rerun_failed=args.rerun_failed, force=args.force)
    logging.info(
        "Processed %d file(s), skipped %d. Archive available at %s",
        result.processed,
        result.skipped,
        result.zip_path,
    )


if __name__ == "__main__":
    main()
