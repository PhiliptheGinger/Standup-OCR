"""Integration helpers for Kraken's CLI tooling."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from dataclasses import dataclass

from PIL import Image, ImageOps

import numpy as np

log = logging.getLogger(__name__)

_PAGE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass
class KrakenSegmentationStats:
    pages: int = 0
    lines: int = 0
    skipped: int = 0
    errors: int = 0


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


def _segment_via_cli(
    image_path: Path,
    model: Optional[str] = None,
    *,
    global_cli_args: Optional[Sequence[str]] = None,
    segment_cli_args: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Call the Kraken CLI to obtain segmentation data as JSON."""

    kraken_cli = shutil.which("kraken")
    if kraken_cli is None:
        raise RuntimeError(
            "The 'kraken' command was not found. Install Kraken with 'pip install kraken[serve]' "
            "and ensure it's available on PATH."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_json = Path(tmpdir) / "segmentation.json"
        cmd: list[str] = [kraken_cli]
        if global_cli_args:
            cmd.extend(global_cli_args)
        cmd.extend(["-i", str(image_path), str(output_json), "binarize"])

        segment_cmd: list[str] = ["segment"]
        if model:
            segment_cmd.extend(["--model", model])
        if segment_cli_args:
            segment_cmd.extend(segment_cli_args)
        cmd.extend(segment_cmd)
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


def _coerce_points(candidate: Any) -> list[tuple[float, float]] | None:
    """Best-effort conversion of Kraken boundary/baseline payloads into points."""

    if candidate is None:
        return None

    if isinstance(candidate, dict):
        if "points" in candidate:
            return _coerce_points(candidate["points"])
        if {"x", "y"}.issubset(candidate.keys()):
            return [(float(candidate["x"]), float(candidate["y"]))]
        numeric_keys = {"x0", "y0", "x1", "y1"}
        if numeric_keys.issubset(candidate.keys()):
            return [
                (float(candidate["x0"]), float(candidate["y0"])),
                (float(candidate["x1"]), float(candidate["y1"])),
            ]
        bbox_keys = {"left", "top", "right", "bottom"}
        if bbox_keys.issubset(candidate.keys()):
            return [
                (float(candidate["left"]), float(candidate["top"])),
                (float(candidate["right"]), float(candidate["bottom"])),
            ]
        return None

    if isinstance(candidate, (list, tuple)):
        if not candidate:
            return None
        first = candidate[0]
        if isinstance(first, (list, tuple, dict)):
            points: list[tuple[float, float]] = []
            for entry in candidate:
                if isinstance(entry, dict):
                    if {"x", "y"}.issubset(entry.keys()):
                        points.append((float(entry["x"]), float(entry["y"])))
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    points.append((float(entry[0]), float(entry[1])))
            return points or None
        if len(candidate) >= 4 and all(isinstance(value, (int, float)) for value in candidate[:4]):
            return [
                (float(candidate[0]), float(candidate[1])),
                (float(candidate[2]), float(candidate[3])),
            ]

    return None


def _bbox_from_points(
    points: list[tuple[float, float]],
    *,
    padding: int = 8,
    extra_vertical: int = 0,
) -> tuple[int, int, int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    left = math.floor(min(xs) - padding)
    top = math.floor(min(ys) - padding - extra_vertical)
    right = math.ceil(max(xs) + padding)
    bottom = math.ceil(max(ys) + padding + extra_vertical)
    return left, top, right, bottom


def _line_bbox(line: Dict[str, Any], padding: int = 8) -> tuple[int, int, int, int] | None:
    bbox_candidate = line.get("bbox")
    points = _coerce_points(bbox_candidate)
    if points:
        return _bbox_from_points(points, padding=padding)

    boundary = _coerce_points(line.get("boundary") or line.get("polygon"))
    if boundary:
        return _bbox_from_points(boundary, padding=padding)

    baseline = _coerce_points(line.get("baseline"))
    if baseline:
        extra = max(padding, 12)
        return _bbox_from_points(baseline, padding=padding, extra_vertical=extra)

    return None


def _clamp_box(
    box: tuple[int, int, int, int], image_size: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = box
    width, height = image_size
    left = max(0, min(width, left))
    top = max(0, min(height, top))
    right = max(0, min(width, right))
    bottom = max(0, min(height, bottom))
    if right - left <= 1 or bottom - top <= 1:
        return None
    return left, top, right, bottom


def _get_segmentation(
    image_path: Path,
    *,
    model: Optional[str] = None,
    out_pagexml: Optional[Path] = None,
    global_cli_args: Optional[Sequence[str]] = None,
    segment_cli_args: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Return Kraken's segmentation dictionary for ``image_path``."""

    _require_kraken()
    segmentation = _segment_via_cli(
        image_path,
        model=model,
        global_cli_args=global_cli_args,
        segment_cli_args=segment_cli_args,
    )

    if not isinstance(segmentation, dict):
        if hasattr(segmentation, "to_dict"):
            segmentation = segmentation.to_dict()
        else:  # pragma: no cover - fallback when API returns a list-like structure
            segmentation = {"lines": segmentation}

    if out_pagexml is not None:
        try:
            from kraken.serialization import serialize  # type: ignore

            xml_bytes = serialize(segmentation=segmentation)
            out_pagexml.write_bytes(xml_bytes)
        except Exception as exc:  # pragma: no cover - serialisation is best-effort
            log.warning("Failed to serialise PAGE-XML via Kraken: %s", exc)

    return segmentation


def segment_lines(
    image_path: Path,
    out_pagexml: Optional[Path] = None,
    *,
    model: Optional[str] = None,
) -> List[List[Tuple[float, float]]]:
    """Run Kraken's baseline segmenter and return a list of baselines."""

    segmentation = _get_segmentation(image_path, model=model, out_pagexml=out_pagexml)

    baselines: List[List[Tuple[float, float]]] = []
    for line in segmentation.get("lines", []):
        baseline = line.get("baseline")
        if not baseline:
            continue
        baselines.append([(float(x), float(y)) for x, y in baseline])

    return baselines


def segment_pages_with_kraken(
    images: Iterable[Path],
    output_dir: Path,
    *,
    model: Optional[str] = None,
    pagexml_dir: Optional[Path] = None,
    padding: int = 12,
    min_width: int = 12,
    min_height: int = 12,
    global_cli_args: Optional[Sequence[str]] = None,
    segment_cli_args: Optional[Sequence[str]] = None,
) -> KrakenSegmentationStats:
    """Segment ``images`` with Kraken and export individual line crops."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pagexml_path: Optional[Path] = None
    if pagexml_dir is not None:
        pagexml_dir = Path(pagexml_dir)
        pagexml_dir.mkdir(parents=True, exist_ok=True)

    stats = KrakenSegmentationStats()

    for image_path in images:
        image_path = Path(image_path)
        if not image_path.exists() or not image_path.is_file():
            log.warning("Skipping %s (not a file)", image_path)
            stats.skipped += 1
            continue
        if image_path.suffix.lower() not in _PAGE_IMAGE_EXTENSIONS:
            log.debug("Skipping %s (unsupported extension)", image_path)
            continue

        stats.pages += 1
        pagexml_path = pagexml_dir / f"{image_path.stem}.xml" if pagexml_dir else None

        try:
            segmentation = _get_segmentation(
                image_path,
                model=model,
                out_pagexml=pagexml_path,
                global_cli_args=global_cli_args,
                segment_cli_args=segment_cli_args,
            )
        except Exception as exc:  # pragma: no cover - runtime Kraken failures
            log.error("Kraken segmentation failed for %s: %s", image_path, exc)
            stats.errors += 1
            continue

        lines = segmentation.get("lines", []) or []
        if not lines:
            log.warning("No lines detected in %s", image_path)
            continue

        try:
            with Image.open(image_path) as im:
                grayscale = im.convert("L")
                width, height = grayscale.size
                for index, line in enumerate(lines, start=1):
                    bbox = _line_bbox(line, padding=padding)
                    if not bbox:
                        stats.skipped += 1
                        continue
                    clamped = _clamp_box(bbox, (width, height))
                    if clamped is None:
                        stats.skipped += 1
                        continue
                    left, top, right, bottom = clamped
                    if (right - left) < min_width or (bottom - top) < min_height:
                        stats.skipped += 1
                        continue

                    crop = grayscale.crop((left, top, right, bottom))
                    out_name = f"{image_path.stem}_line{index:03d}.png"
                    out_image = output_dir / out_name
                    crop.save(out_image)

                    label_path = out_image.with_suffix(".gt.txt")
                    if not label_path.exists():
                        label_path.write_text("", encoding="utf8")

                    metadata = {
                        "source_image": str(image_path),
                        "line_index": index,
                        "bbox": {
                            "left": left,
                            "top": top,
                            "right": right,
                            "bottom": bottom,
                        },
                        "baseline": line.get("baseline"),
                        "boundary": line.get("boundary"),
                        "id": line.get("id"),
                    }
                    metadata_path = out_image.with_suffix(".boxes.json")
                    metadata_path.write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf8",
                    )

                    stats.lines += 1
        except Exception as exc:  # pragma: no cover - image IO failures
            log.error("Failed to crop lines from %s: %s", image_path, exc)
            stats.errors += 1

    return stats


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


def _default_kraken_progress() -> Optional[str]:
    """Return the safest progress renderer for the current platform."""

    if os.name == "nt":
        return "plain"
    return None


def train(
    dataset_dir: Path,
    model_out: Path,
    epochs: int = 50,
    val_split: float = 0.1,
    base_model: Optional[Path] = None,
    progress: Optional[str] = None,
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
        if progress == "none":
            env.pop("KRAKEN_PROGRESS", None)
        elif progress:
            env["KRAKEN_PROGRESS"] = progress
        else:
            default_progress = _default_kraken_progress()
            if default_progress:
                env.setdefault("KRAKEN_PROGRESS", default_progress)

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
