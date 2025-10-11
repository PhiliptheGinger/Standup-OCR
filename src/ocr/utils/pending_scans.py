"""Utilities to report scans that still require transcription."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
READY_FOR_AGENT_DIR = PROJECT_ROOT / "uploads" / "ready_for_agent"
TRANSCRIPTS_RAW_DIR = PROJECT_ROOT / "transcripts" / "raw"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _iter_ready_images() -> Iterable[Path]:
    if not READY_FOR_AGENT_DIR.exists():
        return []
    return [
        path
        for path in sorted(READY_FOR_AGENT_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _transcript_exists(image_path: Path) -> bool:
    target = TRANSCRIPTS_RAW_DIR / f"{image_path.stem}.txt"
    return target.exists()


def find_missing_transcriptions() -> list[str]:
    missing = [path.name for path in _iter_ready_images() if not _transcript_exists(path)]
    return sorted(missing)


def main() -> None:
    missing = find_missing_transcriptions()
    print("Missing transcriptions:")
    if not missing:
        print("(none)")
        return
    for name in missing:
        print(name)


if __name__ == "__main__":
    main()
