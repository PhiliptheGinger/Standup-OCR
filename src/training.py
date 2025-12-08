"""Utilities for fine-tuning a Tesseract LSTM model."""
from __future__ import annotations

import logging
import re
import os
import shutil
import subprocess
import binascii
import logging
from types import SimpleNamespace
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Any, Dict
from dataclasses import dataclass

import cv2
from PIL import Image, ImageDraw, ImageFont

from .preprocessing import preprocess_image
from .gpt_ocr import GPTTranscriber, GPTTranscriptionError

PathLike = str | os.PathLike[str]


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _extract_label(image_path: Path) -> str:
    """Derive the ground-truth text from the training file name."""
    stem = image_path.stem
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[1]:
        label = parts[1]
    else:
        label = parts[0]
    return label.replace("-", " ")


def _bootstrap_sample_training_image(train_dir: Path) -> Path:
    """Create a starter handwriting sample so first-time users can train."""

    sample_path = train_dir / "word_sample.png"
    if sample_path.exists():
        return sample_path

    image_width = 400
    image_height = 160
    image = Image.new("L", (image_width, image_height), color=255)
    draw = ImageDraw.Draw(image)

    text = "sample"
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 72)
    except OSError:  # pragma: no cover - fallback for environments without the font
        font = ImageFont.load_default()

    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except AttributeError:  # pragma: no cover - Pillow < 8 compatibility
        text_width, text_height = draw.textsize(text, font=font)
    x_pos = max((image_width - text_width) // 2, 10)
    y_pos = max((image_height - text_height) // 2, 10)

    draw.text((x_pos, y_pos), text, fill=0, font=font)
    image.save(sample_path, format="PNG")

    return sample_path


def _discover_images(train_dir: Path) -> List[Path]:
    """Find all images with paired .gt.txt files in train_dir and common subdirectories."""
    train_dir.mkdir(parents=True, exist_ok=True)
    supported = lambda p: p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS

    # Scan root
    root_images = [p for p in train_dir.iterdir() if supported(p)]

    # Scan common subdirectories: lines/, images/
    candidate_images = set(root_images)
    for subdir_name in ("lines", "images"):
        subdir = train_dir / subdir_name
        if subdir.exists():
            candidate_images.update(p for p in subdir.rglob("*") if supported(p))

    # ...existing code for filtering by .gt.txt...
    images_with_gt = []
    for img in candidate_images:
        gt_path = img.with_suffix(".gt.txt")
        if gt_path.exists():
            images_with_gt.append(img)
        else:
            logging.warning(
                f"Skipping {img.name}: no paired .gt.txt file"
            )

    return sorted(images_with_gt)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_command(command: Iterable[str], *, cwd: Optional[Path] = None) -> None:
    logging.debug("Running command: %s", " ".join(str(p) for p in command))
    subprocess.run(command, check=True, cwd=cwd)


def _prepare_ground_truth(image_path: Path, label: str, work_dir: Path) -> Tuple[Path, Path]:
    """Write the ground-truth file required by Tesseract training."""
    label = label.strip()
    if not label:
        raise ValueError(f"Empty transcription supplied for {image_path}")
    base_name = image_path.stem
    gt_file = work_dir / f"{base_name}.gt.txt"
    with gt_file.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(label + "\n")

    # Optionally save a preprocessed version alongside the original.
    processed = preprocess_image(image_path)
    processed_path = work_dir / f"{base_name}.png"

    cv2.imwrite(str(processed_path), processed)
    logging.debug("Prepared GT for %s => %s", image_path.name, label)
    return processed_path, gt_file


def _find_existing_lstmf(image_path: Path, work_dir: Path) -> Optional[Path]:
    """
    Check if a valid .lstmf already exists for this image (handles rotated/prepared variants).
    """
    stem = image_path.stem
    candidates = sorted(work_dir.glob(f"{stem}*.lstmf"))
    for candidate in candidates:
        invalid_marker = candidate.with_suffix(".invalid")
        boxes_marker = candidate.with_suffix(".boxes.failed")
        if invalid_marker.exists() or boxes_marker.exists():
            continue
        if _validate_lstmf(candidate):
            return candidate
    return None


def _generate_lstmf(
    image_path: Path,
    work_dir: Path,
    base_lang: str,
    *,
    allow_resume: bool = True,
) -> Optional[Path]:
    """
    Generate a single .lstmf training sample from a line image (resume-aware).
    """
    import logging, subprocess, os
    base_name = image_path.stem
    # Fast resume: if any existing .lstmf matches stem, reuse.
    if allow_resume:
        existing = _find_existing_lstmf(image_path, work_dir)
        if existing:
            logging.info("Reusing existing %s", existing.name)
            return existing
    MAKEBOX_TIMEOUT_S = int(os.environ.get("MAKEBOX_TIMEOUT_S", "15"))
    TRAIN_TIMEOUT_S = int(os.environ.get("TRAIN_TIMEOUT_S", "15"))
    # 1. Prepare (rotate/orient) BEFORE makebox so boxes match final orientation.
    prepared_img = image_path
    try:
        from PIL import Image, ImageOps
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)  # honor EXIF if present
        w, h = img.size
        rotated = False
        if h > w * 1.1:
            # convert portrait to landscape (counterclockwise 90°)
            img = img.rotate(90, expand=True)
            rotated = True
        if rotated or not str(image_path.parent).startswith(str(work_dir)):
            prepared_path = work_dir / f"{base_name}_prepared.png"
            img.save(prepared_path)
            prepared_img = prepared_path
            base_name = prepared_path.stem  # important: align name for box/lstmf
            logging.info("Prepared (rotated=%s) %s -> %s", rotated, image_path.name, prepared_img.name)
    except Exception:
        logging.debug("Orientation step skipped for %s", image_path.name)

    # Ensure ground-truth .gt.txt exists under current (possibly updated) base_name.
    original_gt = image_path.with_suffix(".gt.txt")
    if not original_gt.exists():
        raise FileNotFoundError(f"Ground truth file missing for {image_path.name}: {original_gt}")
    target_gt = work_dir / f"{base_name}.gt.txt"
    if not target_gt.exists():
        gt_content = original_gt.read_text(encoding="utf-8", errors="ignore")
        # Normalize GT: remove tabs, CRs, collapse whitespace, ensure single line
        gt_content = original_gt.read_text(encoding="utf-8", errors="ignore")
        gt_content = gt_content.replace("\t", " ").replace("\r", "").strip()
        # Take only first non-empty line
        gt_lines = [ln.strip() for ln in gt_content.split("\n") if ln.strip()]
        gt_norm = gt_lines[0] if gt_lines else ""
        if not gt_norm:
            raise ValueError(f"Empty ground truth for {image_path.name}")
        target_gt.write_text(gt_norm + "\n", encoding="utf-8")
        logging.debug("Copied GT %s -> %s", original_gt.name, target_gt.relative_to(work_dir))

    output_base = work_dir / base_name
    box_path = output_base.with_suffix(".box")

    # 2. Run makebox on prepared_img (not original).
    psm_candidates = [7, 11, 13, 3, 6]
    makebox_ok = False
    raw_box_lines: list[str] = []
    for psm in psm_candidates:
        cmd = [
            "tesseract",
            str(prepared_img),
            str(output_base),
            "-l",
            base_lang,
            "--psm",
            str(psm),
            "makebox",
        ]
        logging.info("Running makebox (PSM %s): %s", psm, " ".join(cmd))
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=MAKEBOX_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logging.warning("makebox timed out (PSM %s) for %s after %ss; retrying next PSM", psm, prepared_img.name, MAKEBOX_TIMEOUT_S)
            continue
        except subprocess.CalledProcessError as e:
            logging.warning("makebox failed (PSM %s) for %s: %s", psm, prepared_img.name, (e.stderr or "").strip()[:300])
            continue
        if box_path.exists():
            raw_box_lines = [ln.strip() for ln in box_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            # mitigate runaway outputs
            if len(raw_box_lines) > 5000:
                logging.warning("makebox produced excessive lines (%d); capping and falling back", len(raw_box_lines))
                raw_box_lines = []
            if raw_box_lines:
                logging.debug("makebox produced %d lines (PSM %s)", len(raw_box_lines), psm)
                makebox_ok = True
                used_psm = psm
                break
        else:
            err = (result.stderr or "").strip()
            if err:
                logging.debug("makebox stderr (PSM %s): %s", psm, err[:300])

    # 3. Fallback / heuristic recovery (proportional segmentation INCLUDING spaces).
    gt_txt = target_gt  # Use copied/renamed GT file.
    raw_gt = gt_txt.read_text(encoding="utf-8") if gt_txt.exists() else ""
    # Preserve spaces; drop only trailing newlines for counting.
    raw_gt_chars = [c for c in raw_gt if c != "\n" and c != "\r"]
    needed_count = len(raw_gt_chars)

    if needed_count == 0:
        raise RuntimeError(f"No ground-truth characters available for {prepared_img.name}")

    # Attempt smarter subdivision of existing boxes into per-character boxes
    if makebox_ok and raw_box_lines and len(raw_box_lines) < needed_count:
        # Try to split wide/word-level boxes into character boxes first.
        split_result = _split_boxes_to_chars(raw_box_lines, raw_gt)
        if split_result and len(split_result) == needed_count:
            logging.info(
                "Expanded %d makebox entries into %d character boxes for %s",
                len(raw_box_lines),
                len(split_result),
                prepared_img.name,
            )
            raw_box_lines = split_result
            makebox_ok = True
            used_psm = used_psm  # unchanged

    if (not makebox_ok) or (len(raw_box_lines) != needed_count):
        logging.warning(
            "Makebox mismatch for %s: boxes=%d needed=%d. Rebuilding proportional boxes (spaces included).",
            prepared_img.name,
            len(raw_box_lines),
            needed_count,
        )
        try:
            from PIL import Image as PILImage
            im = PILImage.open(prepared_img)
            W, H = im.size
            # Use a vertical band (reduce full-height boxes which Tesseract sometimes rejects)
            text_top = int(H * 0.2)
            text_bottom = int(H * 0.8)
            if text_bottom - text_top < 2:
                # Fallback if height is tiny
                text_top = 0
                text_bottom = max(1, H - 1)
            positions = [round(i * W / needed_count) for i in range(needed_count + 1)]
            recovered: list[str] = []
            for i, ch in enumerate(raw_gt_chars):
                left = positions[i]
                next_pos = positions[i + 1]
                if next_pos <= left:
                    next_pos = left + 1  # ensure progress
                # inclusive right coordinate
                right = min(W - 1, next_pos - 1)
                recovered.append(f"{ch} {left} {text_top} {right} {text_bottom} 0")
            raw_box_lines = recovered
            makebox_ok = True
            used_psm = psm_candidates[0]
            logging.info(
                "Recovered %d box(es) for %s using proportional segmentation (band %d-%d).",
                len(recovered),
                prepared_img.name,
                text_top,
                text_bottom,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to recover boxes for {prepared_img.name} (heuristic error: {e})"
            ) from e

    # 4. Normalize boxes.
    normalized = []
    for ln in raw_box_lines:
        parts = ln.split()
        if len(parts) >= 6:
            ch = parts[0]
            left, bottom, right, top = parts[1:5]
            page = parts[5] if len(parts) >= 6 else "0"
            normalized.append(f"{ch} {left} {bottom} {right} {top} {page}")

    # Validate boxes; rebuild with wider band if invalid (Option A)
    def _validate_boxes(lines: list[str]) -> tuple[bool, dict]:
        stats = {"total": 0, "invalid": 0}
        valid = True
        try:
            from PIL import Image as PILImage
            im = PILImage.open(prepared_img)
            W, H = im.size
        except Exception:
            W = H = None
        for ln in lines:
            stats["total"] += 1
            try:
                ch, l, b, r, t, _pg = ln.split()[0], *ln.split()[1:6]
                l = int(l); b = int(b); r = int(r); t = int(t)
            except Exception:
                stats["invalid"] += 1
                valid = False
                continue
            if r < l or (r - l) < 1 or (t - b) < 2:
                stats["invalid"] += 1
                valid = False
                continue
            if W is not None and H is not None:
                if l < 0 or r >= W or b < 0 or t >= H:
                    stats["invalid"] += 1
                    valid = False
        return valid, stats

    is_valid, stats = _validate_boxes(normalized)
    if not is_valid:
        logging.warning(
            "Skipping %s: invalid box coordinates after rebuild (invalid=%d/%d)",
            prepared_img.name,
            stats["invalid"],
            stats["total"],
        )
        # Optional: drop a small note file for later inspection
        fail_note = work_dir / f"{base_name}.boxes.failed"
        try:
            fail_note.write_text(
                "Invalid boxes after rebuild for "
                f"{prepared_img.name}\n"
                f"invalid={stats['invalid']} total={stats['total']}\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        return None

    # Validate boxes; rebuild with wider band if invalid (Option B)
    def _validate_and_rebuild(lines: list[str], img_path: Path) -> bool:
        try:
            from PIL import Image as PILImage
            im = PILImage.open(img_path)
            W, H = im.size
        except Exception:
            return False
        
        for attempt in range(2):
            invalid_boxes = []
            for ln in lines:
                try:
                    parts = ln.split()
                    l, b, r, t = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                except (ValueError, IndexError):
                    invalid_boxes.append(ln)
                    continue
                
                # Reject invalid geometry
                if r <= l or t <= b:
                    invalid_boxes.append(ln)
                    continue
                if l < 0 or r >= W or b < 0 or t >= H:
                    invalid_boxes.append(ln)
                    continue
                if (r - l) < 1 or (t - b) < 2:
                    invalid_boxes.append(ln)
            
            if not invalid_boxes:
                # All boxes are valid
                return True
            
            # Heuristic: widen the band for the next attempt
            try:
                band = (int(H * 0.1), int(H * 0.9))
                if attempt == 1:
                    band = (0, max(1, H - 1))
                text_top, text_bottom = band
                positions = [round(i * W / needed_count) for i in range(needed_count + 1)]
                rebuilt: list[str] = []
                for i, ch in enumerate(raw_gt_chars):
                    left = positions[i]
                    next_pos = positions[i + 1]
                    if next_pos <= left:
                        next_pos = left + 1
                    right = min(W - 1, next_pos - 1)
                    rebuilt.append(f"{ch} {left} {text_top} {right} {text_bottom} 0")
                normalized = [
                    f"{ln.split()[0]} {ln.split()[1]} {ln.split()[2]} {ln.split()[3]} {ln.split()[4]} 0"
                    for ln in rebuilt
                ]
                logging.info("Rebuilt boxes with band %d-%d: valid=unknown", text_top, text_bottom)
            except Exception:
                logging.warning("Rebuild attempt %d failed for %s", attempt + 1, prepared_img.name)
                continue
        
        return False  # All attempts failed

    if not _validate_and_rebuild(normalized, prepared_img):
        logging.warning("Skipping %s: invalid box coordinates after rebuild attempts", prepared_img.name)
        return None

    box_path.write_text("\n".join(normalized) + "\n", encoding="utf-8")

    # Ensure GT filename matches prepared image basename
    gt_txt_src = image_path.with_suffix(".gt.txt")
    gt_txt_dest = work_dir / f"{base_name}.gt.txt"
    if gt_txt_src.exists():
        if not gt_txt_dest.exists():
            gt_txt_dest.write_text(gt_txt_src.read_text(encoding="utf-8"), encoding="utf-8")

    # Mirror box next to prepared image (Tesseract expects <image>.box)
    image_side_box = prepared_img.with_suffix(".box")
    backup_box = None
    wrote_temp = False
    try:
        if image_side_box.exists():
            backup_box = image_side_box.with_suffix(".box.orig")
            image_side_box.replace(backup_box)
        image_side_box.write_text("\n".join(normalized) + "\n", encoding="utf-8")
        wrote_temp = True
    except Exception as e:
        logging.debug("Could not mirror box beside image %s: %s", prepared_img.name, e)

    # Run lstm.train (capture stderr; retry with safer PSM if needed)
    def _run_train(psm_value: int) -> subprocess.CompletedProcess:
        cmd = [
            "tesseract",
            str(prepared_img),
            str(output_base),
            "--psm",
            str(psm_value),
            "--oem",
            "1",
            "-l",
            base_lang,
            "lstm.train",
        ]
        logging.info("Running training with PSM %s: %s", psm_value, " ".join(cmd))
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=TRAIN_TIMEOUT_S)
        except subprocess.TimeoutExpired as e:
            # fabricate a result-like object with timeout info
            class R(SimpleNamespace): pass
            r = R(returncode=124, stdout="", stderr=f"lstm.train timeout after {TRAIN_TIMEOUT_S}s")
            return r
    try:
        # First attempt with used_psm
        logging.debug("Training inputs: image=%s box=%s gt=%s", str(prepared_img), str(image_side_box), str(target_gt))
        result = _run_train(used_psm)
        lstmf_path = output_base.with_suffix(".lstmf")
        if result.returncode != 0:
            logging.warning("lstm.train failed (PSM %s) for %s: %s",
                            used_psm, prepared_img.name, (result.stderr or "").strip())
        # If non-zero return OR .lstmf missing, retry with alternative PSMs
        if (result.returncode != 0) or (not lstmf_path.exists()):
            for retry_psm in (7, 13, 6, 3):
                if retry_psm == used_psm:
                    continue
                result = _run_train(retry_psm)
                lstmf_path = output_base.with_suffix(".lstmf")
                if result.returncode == 0 and lstmf_path.exists():
                    logging.info("lstm.train succeeded on retry (PSM %s) for %s", retry_psm, prepared_img.name)
                    used_psm = retry_psm
                    break
                else:
                    logging.warning("Retry lstm.train (PSM %s) status=%s exists=%s stderr=%s",
                                    retry_psm, result.returncode, lstmf_path.exists(), (result.stderr or "").strip()[:300])
            # Final check before raising
            if not output_base.with_suffix(".lstmf").exists():
                sample_boxes = "\n".join((normalized[:5] if normalized else []))
                logging.error("Training did not produce .lstmf for %s. Sample boxes:\n%s",
                              prepared_img.name, sample_boxes)
                # mark failure to avoid aborting whole run
                fail_note = work_dir / f"{base_name}.failed"
                try:
                    fail_note.write_text(f"Failed to produce .lstmf for {prepared_img.name}\nstderr={(result.stderr or '').strip()[:500]}\n", encoding="utf-8")
                except Exception:
                    pass
                raise RuntimeError(f"Tesseract training failed for {prepared_img.name}: no .lstmf produced")
    finally:
        # Cleanup temporary image-side .box
        try:
            if wrote_temp:
                if backup_box and backup_box.exists():
                    backup_box.replace(image_side_box)
                else:
                    image_side_box.unlink(missing_ok=True)
        except Exception:
            pass

    lstmf_path = output_base.with_suffix(".lstmf")
    if not lstmf_path.exists():
        raise RuntimeError(f"Expected {lstmf_path.name} not found after training")
    
    # Post-generation validation: size + header + oversize heuristic. Do NOT abort whole run; mark & skip.
    lstmf_size = lstmf_path.stat().st_size
    if not _validate_lstmf(lstmf_path):
        logging.warning("Generated invalid .lstmf for %s (size=%d); will exclude from training", lstmf_path.name, lstmf_size)
        fail_note = work_dir / f"{base_name}.invalid"
        try:
            fail_note.write_text(
                f"Invalid .lstmf (size={lstmf_size}) header/heuristic failed for {prepared_img.name}\n",
                encoding="utf-8"
            )
        except Exception:
            pass
        return None
    logging.info("Generated %s (%d bytes)", lstmf_path.name, lstmf_size)
    return lstmf_path

def _resolve_tessdata_dir(tessdata_dir: Optional[Path]) -> Path:
    """Resolve tessdata directory using explicit path, env, or common locations."""
    import os, shutil, subprocess
    from pathlib import Path

    if tessdata_dir:
        return Path(tessdata_dir)

    env_dir = os.environ.get("TESSDATA_PREFIX")
    if env_dir:
        candidate = Path(env_dir)
        if candidate.exists():
            return candidate

    try:
        result = subprocess.run(
            ["tesseract", "--print-tessdata-dir"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        result = None
    else:
        candidate = Path(result.stdout.strip())
        if candidate.exists():
            return candidate

    exe_path = shutil.which("tesseract")
    if exe_path:
        exe_path = Path(exe_path)
        search_roots = [exe_path.parent, exe_path.parent.parent]
        for root in search_roots:
            candidate = root / "tessdata"
            if candidate.exists():
                return candidate
            share_candidate = root / "share" / "tessdata"
            if share_candidate.exists():
                return share_candidate

    env_roots: list[Path] = []
    for env_var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        path = os.environ.get(env_var)
        if path:
            env_roots.append(Path(path))
    env_roots.append(Path.home())

    known_suffixes = [
        Path("Tesseract-OCR") / "tessdata",
        Path("Programs") / "Tesseract-OCR" / "tessdata",
        Path("scoop") / "apps" / "tesseract" / "current" / "tessdata",
        Path("scoop") / "apps" / "tesseract-nightly" / "current" / "tessdata",
    ]

    for root in env_roots:
        for suffix in known_suffixes:
            candidate = root / suffix
            if candidate.exists():
                return candidate

    linux_defaults = [
        Path("/usr/share/tesseract-ocr/4.00/tessdata"),
        Path("/usr/share/tesseract-ocr/5/tessdata"),
        Path("/usr/share/tesseract-ocr/tessdata"),
        Path("/usr/share/tessdata"),
    ]
    for candidate in linux_defaults:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Unable to locate tessdata directory. Set TESSDATA_PREFIX, install Tesseract, "
        "or pass tessdata_dir explicitly (see README for details)."
    )


def _is_fast_model(base_lang: str, base_traineddata: Path, extracted_dir: Path) -> bool:
    """Return True if the base model is integer-only (fast) and cannot be continued."""

    config_path = extracted_dir / f"{base_lang}.config"
    hints: list[str] = []

    if config_path.exists():
        try:
            hints.append(config_path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            logging.debug("Unable to read %s to inspect int_mode flag", config_path)

    try:
        result = subprocess.run(
            ["combine_tessdata", "-d", str(base_traineddata)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        result = None
    else:
        hints.extend([result.stdout, result.stderr])

    combined_hints = "\n".join(hints).lower()
    fast_tokens = (
        "int_mode 1",
        "int_mode true",
        "modeltype int",
        "network type: int",
        "integer mode",
        "integer (fast) model",
    )
    return any(token in combined_hints for token in fast_tokens)


def _get_unicharset_size(
    base_traineddata: Path, extracted_dir: Path, base_lang: str
) -> Optional[int]:
    """Return the number of characters in the traineddata unicharset if available."""

    # 1) First: try combine_tessdata -d (existing behavior, but with looser patterns)
    try:
        result = subprocess.run(
            ["combine_tessdata", "-d", str(base_traineddata)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        result = None
    else:
        output = (result.stdout or "") + (result.stderr or "")
        # Try several possible phrasings that different Tesseract builds use
        patterns = [
            r"unicharset size:\s*(\d+)",
            r"Unicharset size:\s*(\d+)",
            r"size of unicharset:\s*(\d+)",
            r"unicharset of size\s+(\d+)",
        ]
        for pat in patterns:
            match = re.search(pat, output, re.IGNORECASE)
            if match:
                size = int(match.group(1))
                logging.debug(
                    "Detected unicharset size %s from combine_tessdata output using pattern %r",
                    size,
                    pat,
                )
                return size

    # 2) Second: try to infer size from the extracted .lstm-unicharset file
    unicharset_path = extracted_dir / f"{base_lang}.lstm-unicharset"
    if unicharset_path.exists():
        try:
            text = unicharset_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        if text:
            # Parse lines, ignore blank lines
            lines = [ln for ln in (line.strip() for line in text.splitlines()) if ln]
            # Common formats:
            # - First line is an integer count (N) followed by N entries.
            # - Or the file is just newline-delimited entries (no numeric header).
            if lines:
                first = lines[0]
                if first.isdigit():
                    # Many unicharset files include a leading count. Historically the
                    # count sometimes includes an extra header entry; empirically the
                    # correct number of classes for training is (N - 1) when a numeric
                    # header is present. Use that convention which matches common
                    # combine_tessdata output and avoids the common off-by-one.
                    try:
                        n = int(first)
                        if n >= 1:
                            size = n - 1
                            logging.debug(
                                "Detected unicharset size %s from numeric header in %s",
                                size,
                                unicharset_path,
                            )
                            return size
                    except Exception:
                        pass

                # Fallback: treat each non-empty line as an entry and return the
                # count. If there's a header-like first line, this still returns a
                # sensible value because the header will be counted, and other
                # heuristics (combine_tessdata) are preferred above.
                size = len(lines)
                logging.debug(
                    "Detected unicharset size %s from %s by counting non-empty lines",
                    size,
                    unicharset_path,
                )
                return size

    # 3) Give up: caller should decide whether to error out or fall back
    logging.warning(
        "Could not determine unicharset size for %s; _get_unicharset_size returning None",
        base_traineddata,
    )
    return None


def train_model(
    train_dir: PathLike,
    output_model: str,
    *,
    model_dir: PathLike = "models",
    tessdata_dir: Optional[PathLike] = None,
    base_lang: str = "eng",
    max_iterations: int = 1000,
    unicharset_size_override: Optional[int] = None,
    deserialize_check_limit: Optional[int] = None,
    use_gpt_ocr: bool = True,
    gpt_model: str = "gpt-4o-mini",
    gpt_prompt: Optional[str] = None,
    gpt_cache_dir: Optional[PathLike] = None,
    gpt_max_output_tokens: int = 256,
    gpt_max_images: Optional[int] = None,
    handwriting_ocr: bool = False,
    handwriting_ocr_token: Optional[str] = None,
    resume: bool = True,
) -> Path:
    train_dir = Path(train_dir)
    model_dir = Path(model_dir)
    tessdata_dir = _resolve_tessdata_dir(tessdata_dir)

    # Resolve base_traineddata early (needed for probe step)
    base_traineddata = tessdata_dir / f"{base_lang}.traineddata"
    if not base_traineddata.exists():
        raise FileNotFoundError(
            f"Could not find {base_traineddata}. Install the {base_lang} traineddata "
            "or update `base_lang`."
        )

    images = _discover_images(train_dir)
    work_dir = _ensure_directory(model_dir / f"{output_model}_training")
    logging.info("Starting Tesseract training with %d images (resume=%s)", len(images), resume)

    extracted_dir = work_dir / "extracted"
    _ensure_directory(extracted_dir)

    if unicharset_size_override is not None:
        unicharset_size = unicharset_size_override
        logging.info("Using overridden unicharset size: %d", unicharset_size)
    else:
        unicharset_size = _get_unicharset_size(base_traineddata, extracted_dir, base_lang)
        if unicharset_size == 113:
            logging.warning(
                "Overriding unicharset_size 113 -> 111 for legacy eng.traineddata; "
                "this avoids the 'given outputs 113 not equal to unicharset of 111' error."
            )
            unicharset_size = 111

    probe_net_spec = None
    if unicharset_size is not None:
        probe_net_spec = f"[1,48,0,1 Lfx128 O1c{unicharset_size}]"

    if gpt_max_images is not None:
        if gpt_max_images < 0:
            raise ValueError("gpt_max_images must be zero or a positive integer")

    gpt_transcriptions = 0
    transcriber: Optional[GPTTranscriber] = None
    if use_gpt_ocr:
        transcriber_kwargs: dict[str, object] = {"model": gpt_model, "max_output_tokens": gpt_max_output_tokens}
        if gpt_prompt is not None:
            transcriber_kwargs["prompt"] = gpt_prompt
        if gpt_cache_dir is not None:
            transcriber_kwargs["cache_dir"] = Path(gpt_cache_dir)
        try:
            transcriber = GPTTranscriber(**transcriber_kwargs)
        except GPTTranscriptionError as exc:
            # Don't abort the entire training run if the OpenAI client cannot be
            # initialised (missing/invalid OPENAI_API_KEY). Fall back to no-GPT
            # behaviour so training proceeds with Tesseract-only path.
            logging.warning("ChatGPT OCR disabled: %s. Continuing without GPT OCR.", exc)
            transcriber = None
            use_gpt_ocr = False

    lstmf_paths: List[Path] = []
    gpt_transcriptions = 0
    gpt_limit_reached = False
    for index, image_path in enumerate(images, start=1):
        logging.info("Progress %d/%d (%0.1f%%) -> %s", index, len(images), (index/len(images))*100.0, image_path.name)
        if resume:
            reused = _find_existing_lstmf(image_path, work_dir)
            if reused:
                logging.debug("Skipping %s (already has %s)", image_path.name, reused.name)
                lstmf_paths.append(reused)
                continue
        use_transcriber = False
        if transcriber is not None:
            if gpt_max_images is None or gpt_transcriptions < gpt_max_images:
                use_transcriber = True
            elif not gpt_limit_reached:
                remaining = len(images) - index
                logging.info(
                    "GPT OCR limit of %d image(s) reached; falling back to file-name labels for the remaining %d image(s).",
                    gpt_max_images,
                    remaining,
                )
                gpt_limit_reached = True

        if use_transcriber:
            try:
                label = transcriber.transcribe(image_path)
            except GPTTranscriptionError as exc:
                raise RuntimeError(f"ChatGPT OCR failed for {image_path.name}: {exc}") from exc
            gpt_transcriptions += 1
        else:
            label = _extract_label(image_path)

        processed_path, _ = _prepare_ground_truth(image_path, label, work_dir)
        lstmf_path = _generate_lstmf(
            image_path,
            work_dir,
            base_lang,
            allow_resume=resume,
        )
        if lstmf_path is None:
            continue
        lstmf_paths.append(lstmf_path)

    # Validate .lstmf files and filter out any that fail quick checks.
    valid_lstmf: List[Path] = []
    bad_lstmf: List[Path] = []
    for p in lstmf_paths:
        if _validate_lstmf(p):
            valid_lstmf.append(p)
        else:
            bad_lstmf.append(p)
    if bad_lstmf:
        logging.warning(
            "Excluded %d invalid .lstmf file(s) from training: %s",
            len(bad_lstmf),
            ", ".join(x.name for x in bad_lstmf[:10]),
        )

    # Extract base model components early (needed for probe step)
    extracted_dir = work_dir / "extracted"
    _ensure_directory(extracted_dir)

    combine_prefix = extracted_dir / base_lang
    _run_command([
        "combine_tessdata",
        "-u",
        str(base_traineddata),
        str(combine_prefix),
    ])

    lstm_path = combine_prefix.with_suffix(".lstm")
    if not lstm_path.exists():
        raise RuntimeError("combine_tessdata did not produce the .lstm file")

    if deserialize_check_limit is not None and deserialize_check_limit < 0:
        deserialize_check_limit = None

    total_candidates = len(valid_lstmf)
    if deserialize_check_limit is None or deserialize_check_limit == 0 or deserialize_check_limit >= total_candidates:
        valid_lstmf = _filter_and_fix_lstmf(list(valid_lstmf), base_traineddata, base_lang)
    else:
        to_check = valid_lstmf[:deserialize_check_limit]
        rest = valid_lstmf[deserialize_check_limit:]
        checked_valid = _filter_and_fix_lstmf(list(to_check), base_traineddata, base_lang)
        if rest:
            raise RuntimeError(
                "Deserialize check limit hit before validating all .lstmf files. Rerun with --deserialize-check-limit 0."
            )
        valid_lstmf = checked_valid

    if not valid_lstmf:
        raise RuntimeError(
            "No valid .lstmf training files available after validation. Inspect generated .lstmf files in the training directory."
        )

    probe_artifacts: Optional[ProbeArtifacts] = None
    if probe_net_spec is not None and valid_lstmf:
        try:
            probe_artifacts = _prepare_probe_artifacts(
                base_traineddata,
                work_dir,
                probe_net_spec,
                valid_lstmf[0],
            )
        except RuntimeError as exc:
            logging.warning("Probe bootstrap failed: %s", exc)

    # Dynamic probe step – exclude anything Tesseract itself cannot load.
    probe_ok: List[Path] = []
    probe_bad: List[Path] = []
    if probe_net_spec is None:
        logging.warning("Skipping probe step because unicharset size is unknown; proceeding with %d file(s)", len(valid_lstmf))
        probe_ok = valid_lstmf
    elif probe_artifacts is None:
        logging.warning("Skipping probe step because probe artifacts are unavailable; proceeding with %d file(s)", len(valid_lstmf))
        probe_ok = valid_lstmf
    else:
        for p in valid_lstmf:
            if _probe_lstmf(p, base_traineddata, probe_artifacts):
                probe_ok.append(p)
            else:
                probe_bad.append(p)
        if probe_bad:
            logging.warning("Probe excluded %d file(s): %s",
                            len(probe_bad), ", ".join(x.name for x in probe_bad[:10]))
        if not probe_ok:
            raise RuntimeError("All .lstmf files failed probe – purge and regenerate.")
    list_file = work_dir / "training_files.txt"
    list_file.write_text("\n".join(str(p) for p in probe_ok) + "\n", encoding="utf-8")

    checkpoint_prefix = work_dir / f"{output_model}_checkpoint"
    fast_model = _is_fast_model(base_lang, base_traineddata, extracted_dir)

    def _run_lstmtraining(command: list[str]) -> subprocess.CompletedProcess[str]:
        logging.info("Running: %s", " ".join(str(p) for p in command))
        result = subprocess.run(command, capture_output=True, text=True)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            logging.debug(stdout)
        if stderr:
            logging.debug(stderr)
        return result

    def _ensure_success(result: subprocess.CompletedProcess[str], description: str) -> None:
        if result.returncode == 0:
            return
        output = ((result.stderr or "") + (result.stdout or "")).strip()
        message = output or f"exit code {result.returncode}"
        raise RuntimeError(f"{description} failed: {message}")

    continue_cmd = [
        "lstmtraining",
        "--continue_from",
        str(lstm_path),
        "--model_output",
        str(checkpoint_prefix),
        "--traineddata",
        str(base_traineddata),
        "--train_listfile",
        str(list_file),
        "--max_iterations",
        str(max_iterations),
    ]

    if unicharset_size is None:
        message = (
            "Unable to determine unicharset size from the base traineddata. "
            "Provide a newer traineddata that includes the LSTM unicharset or override the size manually."
        )
        logging.error(message)
        raise RuntimeError(message)

    net_spec = f"[1,48,0,1 Lfx128 O1c{unicharset_size}]"

    scratch_cmd = [
        "lstmtraining",
        "--net_spec",
        net_spec,
        "--model_output",
        str(checkpoint_prefix),
        "--traineddata",
        str(base_traineddata),
        "--train_listfile",
        str(list_file),
        "--max_iterations",
        str(max_iterations),
    ]

    continue_error_tokens = (
        "failed to deserialize lstmtrainer",
        "failed to read continue from network",
        "deserialize header failed",
        "integer (fast) model",
    )

    # emit diagnostics so we can inspect .lstmf contents before training starts
    _debug_lstmf_listing(work_dir)
    if fast_model:
        logging.info(
            "Detected integer-only base model %s; training from scratch.",
            base_traineddata.name,
        )
        result = _run_lstmtraining(scratch_cmd)
        _ensure_success(result, "Training (fresh network)")
    else:
        result = _run_lstmtraining(continue_cmd)
        if result.returncode != 0:
            output = ((result.stderr or "") + (result.stdout or "")).lower()
            if any(token in output for token in continue_error_tokens):
                if "integer (fast) model" in output:
                    logging.warning(
                        "Base model %s is integer-only according to training output; starting from scratch.",
                        lstm_path,
                    )
                else:
                    logging.warning(
                        "Base model %s cannot be continued. Falling back to training from scratch.",
                        lstm_path,
                    )
                result = _run_lstmtraining(scratch_cmd)
                result = _run_lstmtraining(scratch_cmd)
                _ensure_success(result, "Training (fresh network)")
            else:
                _ensure_success(result, "Training")

    checkpoint_file = checkpoint_prefix.with_suffix(".checkpoint")
    if not checkpoint_file.exists():
        raise RuntimeError(
            "Training did not create a checkpoint file. Inspect the logs above for errors."
        )

    final_model = work_dir / f"{output_model}.traineddata"
    _run_command(
        [
            "lstmtraining",
            "--stop_training",
            "--continue_from",
            str(checkpoint_file),
            "--traineddata",
            str(base_traineddata),
            "--model_output",
            str(final_model),
        ]
    )

    target_model = model_dir / f"{output_model}.traineddata"
    target_model.write_bytes(final_model.read_bytes())

    logging.info("Training complete. Model saved to %s", target_model)
    return target_model
def _debug_lstmf_listing(work_dir: Path) -> None:
    """Log sizes and 64-byte heads of .lstmf files in work_dir for troubleshooting."""
    try:
        lstmfs = sorted(work_dir.glob("*.lstmf"))
    except Exception:
        lstmfs = []
    logging.debug("DEBUG: found %d .lstmf files in %s", len(lstmfs), work_dir)
    for p in lstmfs:
        try:
            size = p.stat().st_size
            head = p.open("rb").read(64)
            head_hex = binascii.hexlify(head).decode("ascii")
            logging.debug("  %s size=%d head_hex=%s", p.name, size, head_hex[:160])
        except Exception as e:
            logging.debug("  Could not read %s: %s", p.name, e)


def _split_boxes_to_chars(box_lines: List[str], raw_gt: str) -> Optional[List[str]]:
    """
    Best-effort: split existing box_lines (possibly word-level or clustered boxes)
    into per-character .box lines to match the ground-truth character sequence.

    Strategy:
    - Parse each box line into bbox (left, bottom, right, top).
    - Compute widths and distribute expected characters proportionally across boxes.
    - For each allocation, subdivide the box horizontally and assign successive GT chars.
    Returns list of .box lines or None if parsing fails.
    """
    try:
        gt_chars = [c for c in list(raw_gt.rstrip("\n")) if c not in (" ", "\t", "\n")]
        expected = len(gt_chars)
        if expected == 0 or not box_lines:
            return None

        parsed = []
        widths = []
        for ln in box_lines:
            parts = ln.rsplit(" ", 5)
            if len(parts) != 6:
                # unexpected format; abort
                return None
            label = parts[0]
            # Be permissive with 5 or 6 field formats; tolerate floats.
            try:
                if len(parts) == 6:
                    left, bottom, right, top, page = map(int, parts[1:6])
                else:
                    left, bottom, right, top = map(int, map(round, map(float, parts[1:5])))
                    page = 0
            except Exception:
                # Last attempt: coerce floats then ints for safety
                try:
                    if len(parts) >= 5:
                        left, bottom, right, top = [int(round(float(x))) for x in parts[1:5]]
                        page = int(round(float(parts[5]))) if len(parts) >= 6 else 0
                    else:
                        return None
                except Exception:
                    return None
            w = max(1, right - left)
            parsed.append({"label": label, "l": left, "b": bottom, "r": right, "t": top, "w": w})
            widths.append(w)

        total_w = max(1, sum(widths))
        # initial proportional allocation
        alloc = [max(1, int(round(w / total_w * expected))) for w in widths]
        # adjust to match expected exactly
        cur = sum(alloc)
        idx_order = sorted(range(len(widths)), key=lambda i: widths[i], reverse=True)
        
        # Add characters from widest boxes when under target
        i = 0
        safety = 10000
        while cur < expected and safety > 0:
            alloc[idx_order[i % len(idx_order)]] += 1
            cur += 1
            i += 1
            safety -= 1
        
        # Remove characters from narrowest boxes when over target
        # CRITICAL FIX: abort if we can't decrement without violating min(alloc) >= 1
        i = 0
        safety = 10000
        while cur > expected and safety > 0:
            j = idx_order[-1 - (i % len(idx_order))]
            if alloc[j] > 1:
                alloc[j] -= 1
                cur -= 1
            else:
                # All boxes already at 1 char; can't reduce further.
                # This means sum(widths) < expected chars — abort.
                return None
            i += 1
            safety -= 1
        
        if cur != expected:
            # Safety limit hit or couldn't converge
            return None

        out_lines: List[str] = []
        gi = 0
        for box, n_chars in zip(parsed, alloc):
            l, b, r, t = box["l"], box["b"], box["r"], box["t"]
            width = r - l
            # split width into n_chars cells
            cell_w = max(1, width // n_chars)
            for k in range(n_chars):
                if gi >= expected:
                    break
                cl = l + k * cell_w
                # last cell extend to r to avoid gaps
                cr = r if k == n_chars - 1 else cl + cell_w
                ch = gt_chars[gi]
                out_lines.append(f"{ch} {cl} {b} {cr} {t} 0")
                gi += 1
            if gi >= expected:
                break

        if gi != expected:
            return None
        return out_lines
    except Exception:
        return None


@dataclass
class ProbeArtifacts:
    checkpoint: Path
    traineddata: Path


def _prepare_probe_artifacts(
    base_traineddata: Path,
    work_dir: Path,
    net_spec: Optional[str],
    sample_lstmf: Optional[Path],
) -> Optional[ProbeArtifacts]:
    """Build (or reuse) a tiny checkpoint for probing .lstmf files."""

    if net_spec is None:
        return None

    probe_prefix = work_dir / "_probe_seed"
    checkpoint = probe_prefix.with_suffix(".checkpoint")
    traineddata = probe_prefix.with_suffix(".traineddata")
    if checkpoint.exists() and traineddata.exists():
        return ProbeArtifacts(checkpoint=checkpoint, traineddata=traineddata)

    if sample_lstmf is None:
        logging.warning("Probe bootstrap skipped: no sample .lstmf available")
        return None

    tmp_list = work_dir / "_probe_seed.list"
    tmp_list.write_text(str(sample_lstmf) + "\n", encoding="utf-8")

    try:
        seed_cmd = [
            "lstmtraining",
            "--net_spec",
            net_spec,
            "--model_output",
            str(probe_prefix),
            "--traineddata",
            str(base_traineddata),
            "--train_listfile",
            str(tmp_list),
            "--max_iterations",
            "1",
        ]
        result = subprocess.run(seed_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "probe seed failed").strip()
            raise RuntimeError(message)

        stop_cmd = [
            "lstmtraining",
            "--stop_training",
            "--continue_from",
            str(checkpoint),
            "--traineddata",
            str(base_traineddata),
            "--model_output",
            str(traineddata),
        ]
        stop_result = subprocess.run(stop_cmd, capture_output=True, text=True)
        if stop_result.returncode != 0:
            message = (stop_result.stderr or stop_result.stdout or "probe stop failed").strip()
            raise RuntimeError(message)

        if checkpoint.exists() and traineddata.exists():
            return ProbeArtifacts(checkpoint=checkpoint, traineddata=traineddata)
        raise RuntimeError("Probe artifacts missing after bootstrap")
    finally:
        tmp_list.unlink(missing_ok=True)  # type: ignore[arg-type]

def _probe_lstmf(
    lstmf: Path,
    base_traineddata: Path,
    artifacts: Optional[ProbeArtifacts],
) -> bool:
    """Run a lightweight verification pass so only real .lstmf files reach training."""

    if artifacts is None:
        logging.debug("Probe artifacts unavailable; implicitly accepting %s", lstmf.name)
        return True

    tmp = lstmf.parent / "_single.list"
    probe_prefix = lstmf.parent / f"_probe_{lstmf.stem}"

    def _log_failure(output: str) -> None:
        snippet = output.strip()[:200]
        if "Deserialize header failed" in output or "Load of page 0 failed" in output:
            logging.warning("Probe rejected %s: %s", lstmf.name, snippet)
        else:
            logging.debug("Probe non-zero for %s: %s", lstmf.name, snippet)

    def _cleanup() -> None:
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            for artifact in probe_prefix.parent.glob(f"{probe_prefix.name}*"):
                artifact.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass

    try:
        tmp.write_text(str(lstmf) + "\n", encoding="utf-8")

        # Prefer lstmeval because it loads .lstmf samples without mutating checkpoints.
        if shutil.which("lstmeval"):
            cmd = [
                "lstmeval",
                "--model",
                str(artifacts.traineddata),
                "--traineddata",
                str(base_traineddata),
                "--eval_listfile",
                str(tmp),
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            except subprocess.TimeoutExpired:
                logging.debug("lstmeval probe timeout for %s", lstmf.name)
            except FileNotFoundError:
                # Rare: race where lstmeval disappears between which() check and run.
                pass
            else:
                if r.returncode == 0:
                    return True
                _log_failure((r.stderr or "") + (r.stdout or ""))

        # Fallback: reuse the probe checkpoint so continue_from behaves like training.
        cmd = [
            "lstmtraining",
            "--continue_from",
            str(artifacts.checkpoint),
            "--model_output",
            str(probe_prefix),
            "--traineddata",
            str(base_traineddata),
            "--train_listfile",
            str(tmp),
            "--max_iterations",
            "0",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired:
            logging.debug("lstmtraining probe timeout for %s", lstmf.name)
            return False
        if r.returncode == 0:
            return True
        _log_failure((r.stderr or "") + (r.stdout or ""))
        return False
    except Exception as e:
        logging.debug("Probe error for %s: %s", lstmf.name, e)
        return False
    finally:
        _cleanup()

# Strengthen existing validation (reject tiny + very small old corrupt files)
def _validate_lstmf(lstmf_path: Path) -> bool:
    try:
        if not lstmf_path.exists(): return False
        size = lstmf_path.stat().st_size
        if size < 1024:  # raise minimum size threshold
            logging.warning("Reject tiny .lstmf %s (%d bytes)", lstmf_path.name, size)
            return False
        max_size = int(os.environ.get("LSTMF_MAX_SIZE", "6000000"))
        if size > max_size:
            logging.warning("Reject oversized .lstmf %s (%d > %d)", lstmf_path.name, size, max_size)
            return False
        with lstmf_path.open("rb") as f:
            header = f.read(32)
        if len(header) < 32: return False
        if header.count(header[0]) == len(header):  # uniform bytes
            logging.warning("Uniform header bytes in %s", lstmf_path.name)
            return False
        return True
    except Exception as e:
        logging.debug("Header validation error %s: %s", lstmf_path.name, e)
        return False

def _quick_deserialize_check(lstmf: Path, base_traineddata: Path, timeout: int = 12) -> bool:
    """Ask Tesseract to load a single .lstmf; reject on deserialize failures."""
    tmp_list = lstmf.parent / f"_{lstmf.stem}_single.list"
    probe_prefix = lstmf.parent / f"_{lstmf.stem}_qc"
    failure_tokens = (
        "Deserialize header failed",
        "Load of page 0 failed",
        "Load of images failed",
    )

    def _summarize(output: str) -> str:
        text = output.strip().splitlines()
        return text[0] if text else ""

    def _contains_failure(output: str) -> bool:
        return any(token in output for token in failure_tokens)

    try:
        tmp_list.write_text(str(lstmf) + "\n", encoding="utf-8")

        if shutil.which("lstmeval"):
            cmd = [
                "lstmeval",
                "--model",
                str(base_traineddata),
                "--traineddata",
                str(base_traineddata),
                "--eval_listfile",
                str(tmp_list),
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                logging.debug("lstmeval timeout when checking %s", lstmf.name)
            else:
                if result.returncode == 0:
                    return True
                combined = (result.stderr or "") + (result.stdout or "")
                if _contains_failure(combined):
                    logging.warning("Quick deserialize rejected %s: %s", lstmf.name, _summarize(combined))
                else:
                    logging.debug("lstmeval non-zero for %s: %s", lstmf.name, _summarize(combined))
                return False

        cmd = [
            "lstmtraining",
            "--net_spec",
            "[1,48,0,1 Lfx16 O1c2]",
            "--model_output",
            str(probe_prefix),
            "--traineddata",
            str(base_traineddata),
            "--train_listfile",
            str(tmp_list),
            "--max_iterations",
            "0",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            logging.debug("lstmtraining timeout when checking %s", lstmf.name)
            return False
        if result.returncode == 0:
            return True
        combined = (result.stderr or "") + (result.stdout or "")
        if _contains_failure(combined):
            logging.warning("Quick deserialize rejected %s: %s", lstmf.name, _summarize(combined))
        else:
            logging.debug("lstmtraining non-zero for %s: %s", lstmf.name, _summarize(combined))
        return False
    except Exception as exc:  # pragma: no cover - defensive
        logging.debug("Quick deserialize error for %s: %s", lstmf.name, exc)
        return False
    finally:
        tmp_list.unlink(missing_ok=True)  # type: ignore[arg-type]
        try:
            for artifact in lstmf.parent.glob(f"{probe_prefix.name}*"):
                artifact.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


def _regenerate_lstmf_once(stem_png: Path, base_lang: str) -> bool:
    """Regenerate a single sample: run makebox + lstm.train once. Return True if new .lstmf exists and is >4KB."""
    try:
        stem = stem_png.with_suffix("").name
        out_stem = stem_png.parent / stem
        subprocess.run([
            "tesseract",
            str(stem_png),
            str(out_stem),
            "-l", base_lang,
            "--psm", "7",
            "makebox",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run([
            "tesseract",
            str(stem_png),
            str(out_stem),
            "--psm", "7",
            "--oem", "1",
            "-l", base_lang,
            "lstm.train",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        lstmf = stem_png.with_suffix(".lstmf")
        return lstmf.exists() and lstmf.stat().st_size > 4096
    except Exception:
        return False


def _filter_and_fix_lstmf(candidates: list[Path], base_traineddata: Path, base_lang: str) -> list[Path]:
    """Validate .lstmf via quick deserialize; if it fails, try one regeneration; otherwise mark .invalid and drop."""
    valid: list[Path] = []
    rejected: list[Path] = []
    for p in candidates:
        if p.stat().st_size < 4096:
            rejected.append(p)
            try:
                p.rename(p.with_suffix(".lstmf.invalid"))
            except Exception:
                pass
            continue
        if _quick_deserialize_check(p, base_traineddata):
            valid.append(p)
            continue
        png = p.with_suffix(".png")
        if png.exists() and _regenerate_lstmf_once(png, base_lang):
            if _quick_deserialize_check(p, base_traineddata):
                valid.append(p)
                continue
        rejected.append(p)
        try:
            p.rename(p.with_suffix(".lstmf.invalid"))
        except Exception:
            pass
    if rejected:
        names = ", ".join(x.name for x in rejected[:20])
        logging.warning(
            "Excluded %d .lstmf file(s) failing deserialize: %s%s",
            len(rejected), names, " ..." if len(rejected) > 20 else "",
        )
    return valid

def _sanitize_makebox_lines(makebox_lines: Iterable[str], image_path: Path) -> List[str]:
    with Image.open(image_path) as img:
        width, height = img.size

    sanitized: List[str] = []
    dropped = 0
    for raw in makebox_lines:
        parts = raw.strip().split()
        if len(parts) < 5:
            dropped += 1
            continue

        try:
            left, bottom, right, top = map(int, parts[1:5])
        except ValueError:
            dropped += 1
            continue

        left = max(left, 0)
        bottom = max(bottom, 0)
        right = min(right, width)
        top = min(top, height)

        if right <= left:
            right = min(left + 1, width)
        if top <= bottom:
            top = min(bottom + 1, height)


        sanitized.append(" ".join([parts[0], str(left), str(bottom), str(right), str(top)] + parts[5:]))
    if dropped and sanitized:
        logger.debug("Dropped %d makebox entries for %s while sanitizing", dropped, image_path.name)
    if not sanitized and dropped:
        logger.debug("Sanitization removed all makebox entries for %s", image_path.name)
    return sanitized

def _debug_log_boxes(box_lines: list[str], img_path: Path) -> None:
    try:
        with Image.open(img_path) as im:
            W, H = im.size
    except Exception as e:
        logger.warning("Could not open %s for debug: %s", img_path, e)
        return

    logger.warning("Box debug for %s (W=%d, H=%d):", img_path.name, W, H)
    for ln in box_lines:
        parts = ln.split()
        if len(parts) < 5:
            logger.warning("  BAD LINE: %r", ln)
            continue
        try:
            l, b, r, t = map(int, parts[1:5])
        except ValueError:
            logger.warning("  BAD COORDS: %r", ln)
            continue
        logger.warning("  %s: l=%d, b=%d, r=%d, t=%d", parts[0], l, b, r, t)

    # Debug log what boxes we ended up with for this prepared image
    _debug_log_boxes(normalized, prepared_img)

TRAIN_PSM = 6  # or 7 if each image is a single line

def _run_makebox_for_image(png: Path, base_lang: str, tessdata_dir: str | None) -> Path:
    out_stem = png.with_suffix("")
    cmd = [
        "tesseract",
        str(png),
        str(out_stem),
        "-l", base_lang,
        "--psm", str(TRAIN_PSM),
        "makebox",
    ]
    if tessdata_dir:
        cmd.insert(1, "--tessdata-dir")
        cmd.insert(2, tessdata_dir)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"makebox failed for {png.name}: {result.stdout}")
    box_path = png.with_suffix(".box")
    if not box_path.exists() or box_path.stat().st_size == 0:
        raise RuntimeError(f"Empty .box for {png.name}; segmentation/orientation failed.")
    return box_path

def _run_lstm_train_for_image(png: Path, base_lang: str, tessdata_dir: str | None) -> Path:
    out_stem = png.with_suffix("")
    cmd = [
        "tesseract",
        str(png),
        str(out_stem),
        "--psm", str(TRAIN_PSM),
        "--oem", "1",
        "-l", base_lang,
        "lstm.train",
    ]
    if tessdata_dir:
        cmd.insert(1, "--tessdata-dir")
        cmd.insert(2, tessdata_dir)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"lstm.train failed for {png.name}: {result.stdout}")
    lstmf = png.with_suffix(".lstmf")
    if not lstmf.exists() or lstmf.stat().st_size < 4096:
        raise RuntimeError(f"lstm.train produced no/too-small .lstmf for {png.name}")
    return lstmf

def _find_gt_for_image(stem: Path) -> Path | None:
    """
    Prefer manually annotated ground truth (<stem>.gt.txt) over OCR .txt.
    """
    gt = stem.with_suffix(".gt.txt")
    if gt.exists() and gt.stat().st_size > 0:
        return gt
    txt = stem.with_suffix(".txt")
    if txt.exists() and txt.stat().st_size > 0:
        return txt
    return None
