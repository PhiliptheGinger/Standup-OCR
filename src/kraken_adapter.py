"""Integration helpers for Kraken's CLI tooling."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


def _kraken_exe_in_venv() -> str | None:
    """Return the kraken executable if available on PATH."""

    exe = shutil.which("kraken")
    if exe:
        return exe

    return None


def is_available() -> bool:
    if _kraken_exe_in_venv() is not None:
        return True

    try:
        import kraken  # type: ignore  # pragma: no cover

        return True
    except Exception:  # pragma: no cover - intentionally broad for user feedback
        return False


def _require_kraken() -> None:
    if _kraken_exe_in_venv() is not None:
        return

    if not is_available():
        raise RuntimeError(
            "Kraken is not available. Install it with 'pip install kraken[serve]' "
            "and ensure the 'kraken' and 'ketos' commands are on your PATH."
        )


def _run_cli_segment(image_path: Path) -> dict:
    """Run Kraken's CLI segmenter and return the parsed JSON output."""

    exe = _kraken_exe_in_venv()
    if exe:
        cmd = [exe, "-i", str(image_path), "seg.json", "segment"]
        invocation = "kraken"
    else:
        cmd = ["python", "-m", "kraken", "-i", str(image_path), "seg.json", "segment"]
        invocation = "python -m kraken"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        output_json = tmp_path / "seg.json"
        cmd = [*cmd[:-2], str(output_json), cmd[-1]]
        log.info("Running Kraken segmentation via %s: %s", invocation, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime failure only
            raise RuntimeError(f"Kraken CLI segmentation failed with exit code {exc.returncode}") from exc

        try:
            return json.loads(output_json.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - best effort parsing
            raise RuntimeError(f"Failed to parse Kraken CLI segmentation output: {exc}") from exc


def segment_lines(image_path: Path, out_pagexml: Optional[Path] = None) -> List[List[Tuple[float, float]]]:
    """Run Kraken's baseline segmenter via the CLI and return baselines."""

    _require_kraken()

    segmentation = _run_cli_segment(Path(image_path))

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
            log.warning("Failed to serialise PAGE-XML via Kraken: %s", exc)

    return baselines


def train(
    dataset_dir: Path,
    model_out: Path,
    epochs: int = 50,
    val_split: float = 0.1,
    base_model: Optional[Path] = None,
) -> Path:
    """Call ``ketos train`` with the given dataset directory."""

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

    log.info("Running ketos: %s", " ".join(cmd))
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
    log.info("Running kraken OCR: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(f"kraken executable not found: {exc}") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise RuntimeError(f"kraken ocr failed with exit code {exc.returncode}") from exc
