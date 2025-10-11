"""Validation helpers for the agentic annotation log."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_LOG = PROJECT_ROOT / "annotation.log"
READY_FOR_AGENT_DIR = PROJECT_ROOT / "uploads" / "ready_for_agent"
TRANSCRIPTS_RAW_DIR = PROJECT_ROOT / "transcripts" / "raw"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class LogEntry:
    timestamp: str
    notebook: str
    page: str
    image_name: str
    transcript_path: Path
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


def _parse_line(line: str, line_number: int) -> LogEntry | str:
    parts = line.strip().split()
    if len(parts) < 5:
        return f"Line {line_number}: expected 5 tokens, found {len(parts)}"
    timestamp_token, notebook, page, image_name, transcript = parts[:5]
    if not (timestamp_token.startswith("[") and timestamp_token.endswith("]")):
        return f"Line {line_number}: invalid timestamp token '{timestamp_token}'"
    timestamp = timestamp_token.strip("[]")
    transcript_path = Path(transcript)
    if not transcript_path.is_absolute():
        transcript_path = PROJECT_ROOT / transcript_path
    return LogEntry(
        timestamp=timestamp,
        notebook=notebook,
        page=page,
        image_name=image_name,
        transcript_path=transcript_path,
        line_number=line_number,
    )


def load_log_entries() -> tuple[List[LogEntry], List[str]]:
    if not ANNOTATION_LOG.exists():
        return [], ["annotation.log does not exist"]
    entries: List[LogEntry] = []
    errors: List[str] = []
    with ANNOTATION_LOG.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            parsed = _parse_line(line, idx)
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
        image_counts[entry.image_name] = image_counts.get(entry.image_name, 0) + 1
        transcript_resolved = entry.transcript_path.resolve()
        transcript_counts[transcript_resolved] = transcript_counts.get(transcript_resolved, 0) + 1

        if entry.image_name not in uploads:
            missing_uploads.append(entry.image_name)
        if transcript_resolved not in transcripts and not entry.transcript_path.exists():
            missing_transcripts.append(str(entry.transcript_path))

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
        print("annotation.log is consistent with ready_for_agent/ and transcripts/raw/.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate annotation.log")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate annotation.log against available files",
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
