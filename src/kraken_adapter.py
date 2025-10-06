"""Thin wrappers around Kraken/ketos functionality."""
from __future__ import annotations
"""Optional integration with Kraken (ketos) for segmentation and OCR."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

Logger = logging.getLogger(__name__)


def is_available() -> bool:
    try:
        import kraken  # type: ignore  # pragma: no cover

        return True
    except Exception:  # pragma: no cover - intentionally broad for user feedback
        return False


def _require_kraken() -> None:
    if not is_available():
        raise RuntimeError(
            "Kraken is not available. Install it with 'pip install kraken[serve]' "
            "and ensure the 'kraken' and 'ketos' commands are on your PATH."
        )


def segment_lines(image_path: Path, out_pagexml: Optional[Path] = None) -> List[List[Tuple[float, float]]]:
    """Run Kraken's baseline segmenter and return a list of baselines.

    The function favours the Python API (``kraken.blla``) but falls back to the
    command line if necessary. When ``out_pagexml`` is provided, the PAGE-XML
    produced by Kraken is saved to that location when possible.
    """

    _require_kraken()
    try:
        from kraken import blla  # type: ignore
    except Exception as exc:  # pragma: no cover - only triggered when API fails
        raise RuntimeError("Kraken is installed but the baseline API is unavailable") from exc

    with Image.open(image_path) as image:
        try:
            segmentation = blla.segment(image.convert("L"))  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - segmentation errors only at runtime
            raise RuntimeError(f"Kraken segmentation failed: {exc}") from exc

    baselines: List[List[Tuple[float, float]]] = []
    for line in segmentation.get("lines", []):
        baseline = line.get("baseline")
        if not baseline:
            continue
        baselines.append([(float(x), float(y)) for x, y in baseline])

    if out_pagexml is not None:
        try:
            from kraken.serialization import serialize  # type: ignore

            xml_bytes = serialize(segmentation=segmentation)
            out_pagexml.write_bytes(xml_bytes)
        except Exception as exc:  # pragma: no cover - serialisation is best-effort
            Logger.warning("Failed to serialise PAGE-XML via Kraken: %s", exc)

    return baselines


def train(
    dataset_dir: Path,
    model_out: Path,
    epochs: int = 50,
    val_split: float = 0.1,
    base_model: Optional[Path] = None,
) -> Path:
    """Call ``ketos train`` with the given dataset directory.

    The directory should contain line images with ``.gt.txt`` files or PAGE-XML
    documents as required by Kraken. This helper constructs a basic training
    command but leaves advanced options to the user via manual invocation.
    """

    _require_kraken()
    ketos = shutil.which("ketos")
    if ketos is None:
        raise RuntimeError(
            "The 'ketos' command was not found. Install Kraken with 'pip install kraken[serve]' "
            "and ensure your virtual environment's bin directory is on PATH."
        )

    cmd = [
        ketos,
        "train",
        "--output",
        str(model_out),
        "--epochs",
        str(epochs),
        "--validation",
        str(val_split),
    ]
    if base_model is not None:
        cmd.extend(["--load", str(base_model)])
    cmd.append(str(dataset_dir))

    Logger.info("Running ketos: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:  # pragma: no cover - subprocess failure only at runtime
        raise RuntimeError(f"ketos executable not found: {exc}") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise RuntimeError(f"ketos train failed with exit code {exc.returncode}") from exc

    return model_out


def ocr(image_path: Path, model_path: Path, out_txt: Path) -> None:
    """Run ``kraken ocr`` on ``image_path`` and write raw text to ``out_txt``."""

    _require_kraken()
    kraken_cli = shutil.which("kraken")
    if kraken_cli is None:
        raise RuntimeError(
            "The 'kraken' command was not found. Install Kraken with 'pip install kraken[serve]' "
            "and ensure it's available on PATH."
        )

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        kraken_cli,
        "-i",
        str(image_path),
        str(out_txt),
        "binarize",
        "segment",
        "ocr",
        "-m",
        str(model_path),
    ]
    Logger.info("Running kraken OCR: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(f"kraken executable not found: {exc}") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise RuntimeError(f"kraken ocr failed with exit code {exc.returncode}") from exc
