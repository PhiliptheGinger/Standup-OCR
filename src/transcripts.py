from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

_SUFFIXES: List[str] = [".txt", ".gt.txt"]


def save_transcript(transcripts_dir: Optional[Path], item_path: Path, text: str) -> None:
    """Persist ``text`` alongside the source image if a transcript directory is configured."""

    if not transcripts_dir:
        return
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    target = transcripts_dir / f"{item_path.stem}.txt"
    target.write_text(text, encoding="utf8")


def load_transcript(item_path: Path, transcripts_dir: Optional[Path], *, search_roots: Iterable[Path] | None = None) -> str:
    """Load a transcript for ``item_path`` from known locations.

    Search order:
    1. Explicit transcripts_dir (if provided), for both .txt and .gt.txt.
    2. Sidecar next to the image.
    3. Optional additional roots provided via ``search_roots``.
    Returns the first successfully read UTF-8 file, stripped of trailing newlines.
    """

    candidates: List[Path] = []

    if transcripts_dir is not None:
        for suffix in _SUFFIXES:
            candidates.append(transcripts_dir / f"{item_path.stem}{suffix}")

    for suffix in _SUFFIXES:
        candidates.append(item_path.with_suffix(suffix))

    for root in search_roots or []:
        for suffix in _SUFFIXES:
            candidates.append(Path(root) / f"{item_path.stem}{suffix}")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        try:
            return candidate.read_text(encoding="utf8").rstrip("\n")
        except Exception:
            continue
    return ""
