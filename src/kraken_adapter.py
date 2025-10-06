"""Integration helpers for Kraken's CLI tooling."""

import inspect
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Tuple

Logger = logging.getLogger(__name__)


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


def segment_lines(image_path: Path, out_pagexml: Optional[Path] = None) -> List[List[Tuple[float, float]]]:
    """Run Kraken's baseline segmenter and return a list of baselines.

        try:
            return json.loads(output_json.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - best effort parsing
            raise RuntimeError(f"Failed to parse Kraken CLI segmentation output: {exc}") from exc


def segment_lines(image_path: Path, out_pagexml: Optional[Path] = None) -> List[List[Tuple[float, float]]]:
    """Run Kraken's baseline segmenter via the CLI and return baselines."""

    _require_kraken()
    try:
        segmentation_module = _load_segmentation_module()
    except RuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - only triggered when API fails
        raise RuntimeError("Kraken is installed but the segmentation API is unavailable") from exc

    with Image.open(image_path) as image:
        img = image.convert("L")
        try:
            if hasattr(segmentation_module, "segment"):
                segment_fn = getattr(segmentation_module, "segment")
                kwargs = {}
                signature = inspect.signature(segment_fn)
                if "text_direction" in signature.parameters:
                    kwargs["text_direction"] = "ltr"
                if "script" in signature.parameters:
                    kwargs["script"] = "latin"
                if "model" in signature.parameters:
                    kwargs["model"] = None
                segmentation = segment_fn(img, **kwargs)
            elif hasattr(segmentation_module, "segment_image"):
                segment_fn = getattr(segmentation_module, "segment_image")
                kwargs = {}
                signature = inspect.signature(segment_fn)
                if "model" in signature.parameters:
                    kwargs["model"] = None
                segmentation = segment_fn(img, **kwargs)
            else:  # pragma: no cover - defensive
                raise RuntimeError(
                    "Unsupported Kraken segmentation API: expected 'segment' or 'segment_image'."
                )
        except RuntimeError:
            raise
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
