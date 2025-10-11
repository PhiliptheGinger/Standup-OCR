"""Helper to surface the ready_for_review.zip artefact."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZIP_NAME = "ready_for_review.zip"


def locate_ready_zip() -> Path:
    candidate = PROJECT_ROOT / ZIP_NAME
    if not candidate.exists():
        raise FileNotFoundError(
            f"{ZIP_NAME} was not found. Run 'python -m src.ocr.main --agentic' first."
        )
    return candidate


def describe_zip(path: Path) -> str:
    stat = path.stat()
    timestamp = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    size_mb = stat.st_size / (1024 * 1024)
    return (
        f"Archive: {path.resolve()}\n"
        f"Size: {stat.st_size} bytes ({size_mb:.2f} MiB)\n"
        f"Timestamp: {timestamp}"
    )


def main() -> None:
    try:
        zip_path = locate_ready_zip()
    except FileNotFoundError as exc:
        print(str(exc))
        return
    print(describe_zip(zip_path))


if __name__ == "__main__":
    main()
