"""Integration helpers for Kraken's CLI tooling."""

from __future__ import annotations

import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

import numpy as np

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


@lru_cache(maxsize=None)
def _ketos_train_validation_flag(ketos_path: str) -> str:
    """Return the validation flag supported by ``ketos train``."""

    help_cmd = [ketos_path, "train", "--help"]
    try:
        proc = subprocess.run(
            help_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:  # pragma: no cover - best effort probing
        log.debug("Failed to inspect 'ketos train --help': %s", exc)
        return "--validation"

    help_text = (proc.stdout or "") + (proc.stderr or "")
    if "--partition" in help_text:
        return "--partition"

    return "--validation"


def _discover_ground_truth(dataset_dir: Path) -> list[str]:
    """Return the Kraken ground truth files contained in ``dataset_dir``."""

    if dataset_dir.is_file():
        return [str(dataset_dir)]

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Training data directory {dataset_dir} does not exist")

    def collect(root: Path, suffixes: tuple[str, ...]) -> list[str]:
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        )
        return [str(path) for path in files]

    pagexml_dir = dataset_dir / "pagexml"
    if pagexml_dir.is_dir():
        xml_files = collect(pagexml_dir, (".xml",))
        if xml_files:
            return xml_files

    lines_dir = dataset_dir / "lines"
    if lines_dir.is_dir():
        image_files = collect(lines_dir, (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        if image_files:
            return image_files

    xml_files = collect(dataset_dir, (".xml",))
    if xml_files:
        return xml_files

    image_files = collect(dataset_dir, (".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    if image_files:
        return image_files

    raise RuntimeError(
        "No Kraken ground truth files were found in the training directory. "
        "Export PAGE-XML data or line image crops before running training."
    )


def _label_path_for_line(image_path: Path) -> Path:
    """Return the expected transcription file path for ``image_path``."""

    stem = image_path.stem
    return image_path.with_name(f"{stem}.gt.txt")


def _is_line_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _line_needs_center_padding(image_path: Path, target_height: int = 120) -> bool:
    """Return ``True`` if Kraken's center normalizer is likely to fail."""

    try:
        with Image.open(image_path) as im:
            grayscale = im.convert("L")
            line = np.asarray(grayscale, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - best effort probing only
        log.debug("Skipping center normalization probe for %s: %s", image_path, exc)
        return False

    if line.size == 0:
        return False

    from kraken.lib.lineest import CenterNormalizer  # lazy import

    normalizer = CenterNormalizer(target_height)
    maximum = float(line.max())
    temp = maximum - line
    max_temp = float(temp.max())
    if max_temp > 0:
        temp /= max_temp

    try:
        normalizer.measure(temp)
        normalizer.normalize(line, cval=maximum)
    except ValueError as exc:
        if "inhomogeneous shape" in str(exc):
            return True
        raise

    return False


def _relative_to_dataset(path: Path, dataset_dir: Path) -> Path:
    try:
        return path.relative_to(dataset_dir)
    except ValueError:
        return Path(path.name)


def _copy_transcription(src_image: Path, dst_image: Path) -> None:
    src_label = _label_path_for_line(src_image)
    if not src_label.exists():
        log.debug("No transcription found for %s", src_image)
        return

    dst_label = _label_path_for_line(dst_image)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_label, dst_label)


def _write_padded_line(src: Path, dst: Path, padding: int = 32) -> None:
    with Image.open(src) as im:
        padded = ImageOps.expand(im.convert("L"), border=padding, fill=255)
        dst.parent.mkdir(parents=True, exist_ok=True)
        padded.save(dst)


def _prepare_line_images(
    ground_truth: list[str], dataset_dir: Path, target_height: int = 120
) -> tuple[list[str], tempfile.TemporaryDirectory | None]:
    """Pad problematic line images so Kraken's dewarper stops crashing."""

    if not ground_truth:
        return ground_truth, None

    tmpdir: tempfile.TemporaryDirectory | None = None
    adjusted: list[str] = []
    patched_sources: list[Path] = []

    for entry in ground_truth:
        image_path = Path(entry)
        if not _is_line_image(image_path):
            adjusted.append(entry)
            continue

        try:
            needs_padding = _line_needs_center_padding(image_path, target_height)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Failed to probe %s for center normalization: %s", image_path, exc)
            adjusted.append(entry)
            continue

        if not needs_padding:
            adjusted.append(entry)
            continue

        if tmpdir is None:
            tmpdir = tempfile.TemporaryDirectory(prefix="kraken-lines-")

        dest_root = Path(tmpdir.name)
        rel_path = _relative_to_dataset(image_path, dataset_dir)
        dest_image = dest_root / rel_path

        _write_padded_line(image_path, dest_image)
        _copy_transcription(image_path, dest_image)

        patched_sources.append(image_path)
        adjusted.append(str(dest_image))

    if patched_sources:
        log.warning(
            "Applied additional padding to %d line images to work around Kraken "
            "center normalization failures: %s",
            len(patched_sources),
            ", ".join(str(path) for path in patched_sources),
        )

    return adjusted, tmpdir


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


def _segment_via_cli(image_path: Path) -> Dict[str, Any]:
    """Call the Kraken CLI to obtain segmentation data as JSON."""

    kraken_cli = shutil.which("kraken")
    if kraken_cli is None:
        raise RuntimeError(
            "The 'kraken' command was not found. Install Kraken with 'pip install kraken[serve]' "
            "and ensure it's available on PATH."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "segmentation.json"
        cmd = [
            kraken_cli,
            "segment",
            "-i",
            str(image_path),
            str(output_json),
        ]
        log.info("Running kraken segmentation: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:  # pragma: no cover - subprocess failure only at runtime
            raise RuntimeError(f"kraken executable not found: {exc}") from exc
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            raise RuntimeError(f"kraken segment failed with exit code {exc.returncode}") from exc

        if not output_json.exists():
            raise RuntimeError("Kraken segmentation did not produce any output JSON file.")

        try:
            return json.loads(output_json.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - best effort parsing
            raise RuntimeError(f"Failed to parse Kraken CLI segmentation output: {exc}") from exc


def segment_lines(image_path: Path, out_pagexml: Optional[Path] = None) -> List[List[Tuple[float, float]]]:
    """Run Kraken's baseline segmenter and return a list of baselines."""

    _require_kraken()
    try:
        segmentation_module = _load_segmentation_module()
    except RuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - only triggered when API fails
        raise RuntimeError("Kraken is installed but the segmentation API is unavailable") from exc

    segmentation: Optional[Dict[str, Any]] = None

    with Image.open(image_path) as image:
        img = image.convert("L")
        try:
            if hasattr(segmentation_module, "segment"):
                segment_fn = getattr(segmentation_module, "segment")
                kwargs: Dict[str, Any] = {}
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

    if not isinstance(segmentation, dict):
        if hasattr(segmentation, "to_dict"):
            segmentation = segmentation.to_dict()
        else:  # pragma: no cover - fallback when API returns a list-like structure
            segmentation = {"lines": segmentation}

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


def _run_with_live_output(
    cmd: list[str], *, env: dict[str, str] | None = None
) -> tuple[str, str]:
    """Run ``cmd`` while teeing stdout/stderr to the parent process."""

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def _drain(stream, collector, sink):
        assert stream is not None  # for type checkers
        try:
            for chunk in stream:
                collector.append(chunk)
                sink.write(chunk)
                sink.flush()
        finally:
            stream.close()

    threads: list[threading.Thread] = []
    if process.stdout is not None:
        threads.append(
            threading.Thread(
                target=_drain,
                args=(process.stdout, stdout_chunks, sys.stdout),
                daemon=True,
            )
        )
    if process.stderr is not None:
        threads.append(
            threading.Thread(
                target=_drain,
                args=(process.stderr, stderr_chunks, sys.stderr),
                daemon=True,
            )
        )

    for thread in threads:
        thread.start()

    returncode = process.wait()

    for thread in threads:
        thread.join()

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)

    if returncode != 0:
        raise subprocess.CalledProcessError(
            returncode,
            cmd,
            output=stdout,
            stderr=stderr,
        )

    return stdout, stderr


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

    ground_truth = _discover_ground_truth(dataset_dir)
    sanitized_tmp: tempfile.TemporaryDirectory | None = None
    try:
        ground_truth, sanitized_tmp = _prepare_line_images(ground_truth, dataset_dir)

        cmd = [
            ketos,
            "train",
            "--output",
            str(model_out),
            "--epochs",
            str(epochs),
        ]
        if val_split > 0:
            validation_flag = _ketos_train_validation_flag(ketos)
            cmd.extend([validation_flag, str(val_split)])
        if base_model is not None:
            cmd.extend(["--load", str(base_model)])
        cmd.extend(ground_truth)

        log.info("Running ketos: %s", " ".join(cmd))
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")

        try:
            _run_with_live_output(cmd, env=env)
        except FileNotFoundError as exc:  # pragma: no cover - subprocess failure only at runtime
            raise RuntimeError(f"ketos executable not found: {exc}") from exc
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            stderr = (exc.stderr or "").strip()
            stdout = getattr(exc, "stdout", "") or getattr(exc, "output", "")
            combined_output = "\n".join(
                part for part in [stdout.strip(), stderr] if part
            )

            if "Model did not improve during" in combined_output:
                hint = (
                    "Kraken aborted training because the validation metric never improved. "
                    "Add more line images or adjust the validation split/epoch count before retrying."
                )
                raise RuntimeError(f"{hint} (ketos exit code {exc.returncode})") from exc

            if combined_output and not stderr:
                stderr = combined_output

            if stderr:
                raise RuntimeError(
                    f"ketos train failed with exit code {exc.returncode}: {stderr}"
                ) from exc

            raise RuntimeError(f"ketos train failed with exit code {exc.returncode}") from exc
    finally:
        if sanitized_tmp is not None:
            sanitized_tmp.cleanup()

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


def ocr_to_string(image_path: Path, model_path: Path) -> str:
    """Run Kraken OCR on ``image_path`` and return the recognised text."""

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
        out_path = Path(handle.name)

    try:
        ocr(image_path, model_path, out_path)
        try:
            return out_path.read_text(encoding="utf8").strip()
        except OSError as exc:  # pragma: no cover - best effort read
            log.debug("Failed to read Kraken OCR output %s: %s", out_path, exc)
            return ""
    finally:
        try:
            out_path.unlink()
        except OSError:  # pragma: no cover - file may already be gone
            pass
