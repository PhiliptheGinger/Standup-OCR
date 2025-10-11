"""Background watcher that triggers agentic OCR when new scans arrive."""
from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..ocr.main import (
    PROJECT_ROOT,
    READY_FOR_AGENT_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
    agentic_batch_transcribe,
)

LOG_PATH = PROJECT_ROOT / "agent_activity.log"


class DebouncedRunner:
    """Run ``callback`` after a quiet period to coalesce rapid file events."""

    def __init__(self, callback: Callable[[], None], delay: float = 1.5) -> None:
        self._callback = callback
        self._delay = delay
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()

    def trigger(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self) -> None:
        try:
            self._callback()
        finally:
            with self._lock:
                self._timer = None


class ReadyFolderHandler(FileSystemEventHandler):
    """Watchdog handler that reacts to new scan files."""

    def __init__(self, on_change: Callable[[], None]) -> None:
        super().__init__()
        self._on_change = on_change

    def on_created(self, event: FileSystemEvent) -> None:  # noqa: D401 - watchdog API
        self._handle(event)

    def on_moved(self, event: FileSystemEvent) -> None:  # noqa: D401 - watchdog API
        self._handle(event)

    def _handle(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(getattr(event, "dest_path", event.src_path))
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            logging.info("Detected new scan: %s", path.name)
            self._on_change()


def _run_agentic_once() -> None:
    try:
        result = agentic_batch_transcribe()
    except Exception as exc:  # pragma: no cover - runtime logging only
        logging.exception("Agentic OCR failed: %s", exc)
        return
    logging.info(
        "Agentic OCR complete: processed=%d skipped=%d archive=%s",
        result.processed,
        result.skipped,
        result.zip_path,
    )


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.FileHandler(LOG_PATH), logging.StreamHandler()]
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Watch uploads/ready_for_agent for new scans")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    LOG_PATH.touch(exist_ok=True)
    configure_logging(verbose=args.verbose)

    READY_FOR_AGENT_DIR.mkdir(parents=True, exist_ok=True)

    runner = DebouncedRunner(_run_agentic_once)
    handler = ReadyFolderHandler(runner.trigger)
    observer = Observer()
    observer.schedule(handler, str(READY_FOR_AGENT_DIR), recursive=False)
    observer.start()
    logging.info("Watching %s for new scans...", READY_FOR_AGENT_DIR)

    # Initial pass to process any pending files.
    _run_agentic_once()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:  # pragma: no cover - manual interruption
        logging.info("Shutting down watchdog...")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
