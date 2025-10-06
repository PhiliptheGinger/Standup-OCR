"""Thin wrappers around Kraken/ketos functionality."""
from __future__ import annotations
"""Optional integration with Kraken (ketos) for segmentation and OCR."""

import inspect
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType
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


def _explain_import_error(exc: ImportError, previous_exc: ImportError | None = None) -> str:
    """Return a user-facing explanation for Kraken import errors."""

    message = str(exc)
    if "kraken.blla is not a package" in message or "'kraken' is not a package" in message:
        return (
            "This Kraken installation no longer exposes the legacy 'kraken.blla' module. "
            "Recent releases moved the segmentation API to 'kraken.lib.segmentation'. "
            "Upgrade Standup-OCR or reinstall Kraken with 'pip install -U kraken[serve]' "
            "to ensure the new module is available."
        )

    if previous_exc is not None:
        previous_name = getattr(previous_exc, "name", "")
        if previous_name == "kraken.lib.segmentation":
            return (
                "Kraken's modern segmentation API ('kraken.lib.segmentation') could not "
                "be imported, and the legacy fallback also failed. Upgrade Kraken with "
                "'pip install -U kraken[serve]' to obtain a compatible release."
            )

    if "not a package" in message:
        return (
            "Kraken could not be imported because another module named 'kraken' "
            "is shadowing the official library. Rename or remove the conflicting "
            "module (for example a local 'kraken.py' file) and reinstall the "
            "package with 'pip install -U kraken[serve]'."
        )

    missing = getattr(exc, "name", "")
    if missing in {"kraken", "kraken.blla", "kraken.lib.segmentation"}:
        return (
            "Kraken's Python API is unavailable. Ensure it is installed in the "
            "current environment by running 'pip install -U kraken[serve]'."
        )

    return f"Kraken is installed but failed to load its segmentation API ({message})."


def _load_segmentation_module() -> ModuleType:
    """Load Kraken's segmentation module with support for multiple versions."""

    lib_exc: ImportError | None = None
    try:
        from kraken.lib import segmentation as segmentation_mod  # type: ignore

        return segmentation_mod
    except ImportError as exc:
        lib_exc = exc

    try:
        from kraken import blla as segmentation_mod  # type: ignore

        return segmentation_mod
    except ImportError as exc:
        raise RuntimeError(_explain_import_error(exc, lib_exc)) from exc


def _segment_via_cli(image_path: Path) -> dict:
    """Run Kraken's CLI segmenter and return the parsed JSON output."""

    kraken_cmd = shutil.which("kraken")
    if kraken_cmd is None:
        raise RuntimeError(
            "Kraken's Python API is unavailable and the 'kraken' CLI could not be found. "
            "Install Kraken with 'pip install -U kraken[serve]' to obtain the CLI tools."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "segmentation.json"
        cmd = [kraken_cmd, "-i", str(image_path), str(output_json), "segment"]
        Logger.info("Falling back to Kraken CLI segmentation: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime failure only
            raise RuntimeError(f"Kraken CLI segmentation failed with exit code {exc.returncode}") from exc

        try:
            return json.loads(output_json.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - best effort parsing
            raise RuntimeError(f"Failed to parse Kraken CLI segmentation output: {exc}") from exc


def segment_lines(image_path: Path, out_pagexml: Optional[Path] = None) -> List[List[Tuple[float, float]]]:
    """Run Kraken's baseline segmenter and return a list of baselines.

    The function favours the Python API (``kraken.blla``) but falls back to the
    command line if necessary. When ``out_pagexml`` is provided, the PAGE-XML
    produced by Kraken is saved to that location when possible.
    """

    _require_kraken()
    try:
        segmentation_module = _load_segmentation_module()
    except RuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - only triggered when API fails
        raise RuntimeError("Kraken is installed but the segmentation API is unavailable") from exc

    with Image.open(image_path) as image:
        img = image.convert("L")
        segmentation = None

        def _call_segment(function_name: str, extra_kwargs: dict[str, object]) -> Optional[dict]:
            fn = getattr(segmentation_module, function_name, None)
            if fn is None or not callable(fn):
                return None

            kwargs = {}
            signature = inspect.signature(fn)
            for key, value in extra_kwargs.items():
                if key in signature.parameters:
                    kwargs[key] = value

            try:
                return fn(img, **kwargs)
            except TypeError:  # pragma: no cover - signature mismatch, try other APIs
                return None

        try:
            segmentation = _call_segment(
                "segment",
                {"text_direction": "ltr", "script": "latin", "model": None},
            )

            if segmentation is None:
                segmentation = _call_segment("segment_image", {"model": None})

            if segmentation is None:
                segmentation = _call_segment("baseline_segment", {"text_direction": "ltr", "model": None})

            if segmentation is None:
                segmenter_cls = getattr(segmentation_module, "Segmenter", None)
                if segmenter_cls is not None:
                    try:
                        segmenter = segmenter_cls()
                        if hasattr(segmenter, "segment") and callable(segmenter.segment):
                            segmentation = segmenter.segment(img)
                        elif callable(segmenter):
                            segmentation = segmenter(img)
                    except TypeError:  # pragma: no cover - requires args we don't know
                        pass

            if segmentation is None:
                segmentation = _call_segment("extract_baselines", {})

        except Exception as exc:  # pragma: no cover - segmentation errors only at runtime
            raise RuntimeError(f"Kraken segmentation failed: {exc}") from exc

    if segmentation is None:
        segmentation = _segment_via_cli(image_path)

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
