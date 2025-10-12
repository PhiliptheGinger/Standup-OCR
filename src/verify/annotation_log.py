"""Validation helpers for the agentic annotation log."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_LOG = PROJECT_ROOT / "train" / "annotation_log.csv"
READY_FOR_AGENT_DIR = PROJECT_ROOT / "uploads" / "ready_for_agent"
TRANSCRIPTS_RAW_DIR = PROJECT_ROOT / "transcripts" / "raw"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class LogEntry:
    page: str
    transcription: str
    timestamp: str
    line_number: int


@dataclass
class LogValidation:
    corrupt_entries: List[str]
    missing_uploads: List[str]
    missing_transcripts: List[str]
    duplicate_images: List[str]
    duplicate_transcripts: List[str]
    unlogged_uploads: List[str]
    unlogged_transcripts: List[str]


def _iter_uploads() -> Iterable[Path]:
    if not READY_FOR_AGENT_DIR.exists():
        return []
    return [
        path
        for path in READY_FOR_AGENT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _iter_transcripts() -> Iterable[Path]:
    if not TRANSCRIPTS_RAW_DIR.exists():
        return []
    return [
        path
        for path in TRANSCRIPTS_RAW_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    ]


def _parse_row(row: dict[str, str], line_number: int) -> LogEntry | str:
    missing = [key for key in ("page", "transcription", "timestamp") if key not in row]
    if missing:
        return f"Line {line_number}: missing column(s) {', '.join(missing)}"
    page = row["page"].strip()
    transcription = row["transcription"].strip()
    timestamp = row["timestamp"].strip()
    if not page:
        return f"Line {line_number}: empty page value"
    if not timestamp:
        return f"Line {line_number}: empty timestamp value"
    return LogEntry(
        page=page,
        transcription=transcription,
        timestamp=timestamp,
        line_number=line_number,
    )


def load_log_entries() -> tuple[List[LogEntry], List[str]]:
    if not ANNOTATION_LOG.exists():
        return [], ["train/annotation_log.csv does not exist"]
    entries: List[LogEntry] = []
    errors: List[str] = []
    with ANNOTATION_LOG.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        expected = {"page", "transcription", "timestamp"}
        if reader.fieldnames is None:
            return [], ["annotation_log.csv is empty"]
        missing_columns = expected - set(reader.fieldnames)
        if missing_columns:
            errors.append(
                "annotation_log.csv missing columns: " + ", ".join(sorted(missing_columns))
            )
        for idx, row in enumerate(reader, start=2):
            parsed = _parse_row(row, idx)
            if isinstance(parsed, str):
                errors.append(parsed)
            else:
                entries.append(parsed)
    return entries, errors


def validate_entries(entries: List[LogEntry]) -> LogValidation:
    uploads = {path.name for path in _iter_uploads()}
    transcripts = {path.resolve() for path in _iter_transcripts()}

    image_counts: dict[str, int] = {}
    transcript_counts: dict[Path, int] = {}
    missing_uploads: List[str] = []
    missing_transcripts: List[str] = []

    for entry in entries:
        page_name = entry.page
        image_counts[page_name] = image_counts.get(page_name, 0) + 1
        transcript_path = TRANSCRIPTS_RAW_DIR / f"{Path(page_name).stem}.txt"
        transcript_resolved = transcript_path.resolve()
        transcript_counts[transcript_resolved] = transcript_counts.get(transcript_resolved, 0) + 1

        if page_name not in uploads:
            missing_uploads.append(page_name)
        if transcript_resolved not in transcripts and not transcript_path.exists():
            missing_transcripts.append(str(transcript_path))

    logged_images = set(image_counts.keys())
    logged_transcripts = set(transcript_counts.keys())

    duplicate_images = sorted(name for name, count in image_counts.items() if count > 1)
    duplicate_transcripts = sorted(
        str(path) for path, count in transcript_counts.items() if count > 1
    )
    unlogged_uploads = sorted(uploads - logged_images)
    unlogged_transcripts = sorted(
        str(path) for path in transcripts - logged_transcripts
    )

    return LogValidation(
        corrupt_entries=[],
        missing_uploads=sorted(set(missing_uploads)),
        missing_transcripts=sorted(set(missing_transcripts)),
        duplicate_images=duplicate_images,
        duplicate_transcripts=duplicate_transcripts,
        unlogged_uploads=unlogged_uploads,
        unlogged_transcripts=unlogged_transcripts,
    )


def run_check() -> None:
    entries, errors = load_log_entries()
    validation = validate_entries(entries)
    validation.corrupt_entries.extend(errors)

    issues_found = False

    if validation.corrupt_entries:
        issues_found = True
        print("Corrupt entries:")
        for message in validation.corrupt_entries:
            print(f"  - {message}")
    if validation.missing_uploads:
        issues_found = True
        print("Missing uploads referenced in log:")
        for name in validation.missing_uploads:
            print(f"  - {name}")
    if validation.missing_transcripts:
        issues_found = True
        print("Missing transcript files:")
        for path in validation.missing_transcripts:
            print(f"  - {path}")
    if validation.duplicate_images:
        issues_found = True
        print("Duplicate log entries (image names):")
        for name in validation.duplicate_images:
            print(f"  - {name}")
    if validation.duplicate_transcripts:
        issues_found = True
        print("Duplicate log entries (transcript paths):")
        for path in validation.duplicate_transcripts:
            print(f"  - {path}")
    if validation.unlogged_uploads:
        issues_found = True
        print("Uploads without a matching log entry:")
        for name in validation.unlogged_uploads:
            print(f"  - {name}")
    if validation.unlogged_transcripts:
        issues_found = True
        print("Transcript files without a matching log entry:")
        for path in validation.unlogged_transcripts:
            print(f"  - {path}")

    if not issues_found:
        print(
            "annotation_log.csv is consistent with ready_for_agent/ and transcripts/raw/."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate annotation_log.csv")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate annotation_log.csv against available files",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.check:
        parser.print_help()
        return
    run_check()


if __name__ == "__main__":
    main()
