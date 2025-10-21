"""Utilities for fine-tuning a Tesseract LSTM model."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

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
    """Return a sorted list of image paths inside ``train_dir``."""

    train_dir.mkdir(parents=True, exist_ok=True)
    supported = (
        lambda p: p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    root_images = [p for p in train_dir.iterdir() if supported(p)]

    lines_dir = train_dir / "lines"
    nested_images: list[Path] = []
    if lines_dir.exists():
        nested_images = [p for p in lines_dir.rglob("*") if supported(p)]

    candidate_images = set(root_images)
    candidate_images.update(nested_images)

    if candidate_images:
        sample_path = train_dir / "word_sample.png"
        if sample_path in candidate_images and len(candidate_images) > 1:
            candidate_images.remove(sample_path)
        return sorted(candidate_images)

    lines_dir = train_dir / "lines"
    if lines_dir.exists():
        nested_images = [
            p
            for p in sorted(lines_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if nested_images:
            return nested_images

    sample = _bootstrap_sample_training_image(train_dir)
    logging.info(
        "No training images found in %s; created starter sample %s. Replace this image with your own handwriting to continue training.",
        train_dir,
        sample.name,
    )
    return [sample]


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


def _generate_lstmf(image_path: Path, work_dir: Path, base_lang: str) -> Path:
    """Generate .lstmf file for a single training pair.
    Works with either .gt.txt or .box training files."""

    image_path = Path(image_path)
    base_name = image_path.stem
    gt_txt = image_path.with_suffix(".gt.txt")
    lstmf_path = work_dir / f"{base_name}.lstmf"
    box_path = work_dir / f"{base_name}.box"

    if lstmf_path.exists():
        return lstmf_path  # skip if already generated

    if not gt_txt.exists():
        raise FileNotFoundError(f"Missing ground truth file for {image_path}")

    makebox_cmd = [
        "tesseract",
        str(image_path),
        str(work_dir / base_name),
        "-l",
        base_lang,
        "--psm",
        "7",
        "makebox",
    ]

    logging.info("Running: %s", " ".join(makebox_cmd))
    try:
        subprocess.run(makebox_cmd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - surface context
        output = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"Tesseract makebox failed for {image_path}: {output or exc.returncode}"
        ) from exc

    if not box_path.exists():
        raise RuntimeError(
            f"Tesseract did not produce {box_path}. Check that the training tools are installed."
        )

    gt_text = gt_txt.read_text(encoding="utf-8").replace("\r", "").rstrip("\n")
    box_lines = box_path.read_text(encoding="utf-8").splitlines()
    if len(box_lines) < len(gt_text):
        raise RuntimeError(
            f"Ground truth for {image_path.name} has {len(gt_text)} characters but the generated box file contains {len(box_lines)} entries."
        )

    trimmed_lines = box_lines[: len(gt_text)]
    fixed_lines: list[str] = []
    for character, line in zip(gt_text, trimmed_lines):
        parts = line.rsplit(" ", 5)
        if len(parts) != 6:
            raise RuntimeError(
                f"Unexpected box format for {box_path}: '{line}'. Tesseract produced an invalid entry."
            )
        _, left, bottom, right, top, page = parts
        normalized_character = "" if character == "\n" else character
        fixed_lines.append(
            f"{normalized_character} {left} {bottom} {right} {top} {page}"
        )

    if len(box_lines) > len(gt_text):
        logging.info(
            "Discarded %d extra box entries for %s to match the ground truth length.",
            len(box_lines) - len(gt_text),
            image_path.name,
        )

    box_path.write_text("\n".join(fixed_lines) + "\n", encoding="utf-8")

    cmd = [
        "tesseract",
        str(image_path),
        str(work_dir / base_name),
        "--psm",
        "6",
        "--oem",
        "1",
        "-l",
        base_lang,
        "lstm.train",
    ]

    logging.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - surface context
        output = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            f"Tesseract training failed for {image_path}: {output or exc.returncode}"
        ) from exc

    if not lstmf_path.exists():
        exe_path = shutil.which("tesseract")
        if exe_path:
            abs_cmd = [exe_path, *cmd[1:]]
            logging.info(
                "Retrying with absolute Tesseract path: %s", " ".join(abs_cmd)
            )
            try:
                subprocess.run(abs_cmd, check=True)
            except subprocess.CalledProcessError as exc:  # pragma: no cover
                output = (exc.stderr or exc.stdout or "").strip()
                raise RuntimeError(
                    f"Tesseract training failed for {image_path} when using {exe_path}: {output or exc.returncode}"
                ) from exc

    if not lstmf_path.exists():
        raise RuntimeError(
            f"Tesseract did not produce {lstmf_path}. Ensure the language data and training tools are installed."
        )
    return lstmf_path


def _resolve_tessdata_dir(tessdata_dir: Optional[PathLike]) -> Path:
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
            if not root:
                continue
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
    )
    return any(token in combined_hints for token in fast_tokens)


def train_model(
    train_dir: PathLike,
    output_model: str,
    *,
    model_dir: PathLike = "models",
    tessdata_dir: Optional[PathLike] = None,
    base_lang: str = "eng",
    max_iterations: int = 1000,
    use_gpt_ocr: bool = True,
    gpt_model: str = "gpt-4o-mini",
    gpt_prompt: Optional[str] = None,
    gpt_cache_dir: Optional[PathLike] = None,
    gpt_max_output_tokens: int = 256,
    gpt_max_images: Optional[int] = None,
) -> Path:
    """Fine-tune a Tesseract model using handwriting samples.

    This function automates the LSTM training workflow. It expects the
    ``train_dir`` folder to contain images of your handwriting. The image file
    name must encode the correct transcription, for example ``word_hello.png``
    or ``char_A.png``. The portion after the first underscore (``hello`` or
    ``A`` in the examples) is treated as the text shown in the image.

    Parameters
    ----------
    train_dir:
        Directory that holds your handwriting images (place them under
        ``train/`` in this repository).
    output_model:
        Base file name for the resulting ``.traineddata`` model. The file will
        be saved inside ``model_dir``.
    model_dir:
        Directory where training artefacts and the final model should be stored.
    tessdata_dir:
        Location of the base language ``.traineddata`` files. Defaults to the
        ``TESSDATA_PREFIX`` environment variable or the typical Linux install
        path.
    base_lang:
        The language code used as a starting point, defaults to ``eng``.
    max_iterations:
        How many training iterations to run. Increase this value when you add
        more samples.
    use_gpt_ocr:
        When ``True`` (default) each training image is transcribed with
        ChatGPT's vision API to derive the ground-truth label. Disable this to
        fall back to file-name derived labels.
    gpt_model:
        The ChatGPT model identifier to call when ``use_gpt_ocr`` is enabled.
    gpt_prompt:
        Optional custom prompt to send alongside each image when requesting a
        transcription.
    gpt_cache_dir:
        Optional directory where ChatGPT transcriptions are cached. Cached
        files are reused on subsequent runs to avoid repeated API calls.
    gpt_max_output_tokens:
        Maximum number of tokens ChatGPT may return for each transcription
        request.
    gpt_max_images:
        Optional upper bound on how many images should be transcribed with
        ChatGPT. Once the limit is reached the remaining samples fall back to
        file-name derived labels so you can cap API usage.

    Returns
    -------
    Path
        Path to the newly created ``.traineddata`` file.
    """

    train_dir = Path(train_dir)
    model_dir = Path(model_dir)
    tessdata_dir = _resolve_tessdata_dir(tessdata_dir)

    images = _discover_images(train_dir)
    work_dir = _ensure_directory(model_dir / f"{output_model}_training")

    logging.info("Starting Tesseract training with %d images", len(images))

    if gpt_max_images is not None:
        if gpt_max_images < 0:
            raise ValueError("gpt_max_images must be zero or a positive integer")

    transcriber: Optional[GPTTranscriber] = None
    if use_gpt_ocr and (gpt_max_images is None or gpt_max_images != 0):
        transcriber_kwargs: dict[str, object] = {"model": gpt_model, "max_output_tokens": gpt_max_output_tokens}
        if gpt_prompt is not None:
            transcriber_kwargs["prompt"] = gpt_prompt
        if gpt_cache_dir is not None:
            transcriber_kwargs["cache_dir"] = Path(gpt_cache_dir)
        try:
            transcriber = GPTTranscriber(**transcriber_kwargs)
        except GPTTranscriptionError as exc:
            raise RuntimeError(f"Unable to initialise ChatGPT OCR: {exc}") from exc

    lstmf_paths: List[Path] = []
    gpt_transcriptions = 0
    gpt_limit_reached = False
    for index, image_path in enumerate(images):
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
        lstmf_path = _generate_lstmf(processed_path, work_dir, base_lang)
        lstmf_paths.append(lstmf_path)

    list_file = work_dir / "training_files.txt"
    list_file.write_text("\n".join(str(p) for p in lstmf_paths) + "\n", encoding="utf-8")

    base_traineddata = tessdata_dir / f"{base_lang}.traineddata"
    if not base_traineddata.exists():
        raise FileNotFoundError(
            f"Could not find {base_traineddata}. Install the {base_lang} traineddata "
            "or update `base_lang`."
        )

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

    scratch_cmd = [
        "lstmtraining",
        "--net_spec",
        "[1,48,0,1 Lfx128 O1c1]",
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
    )

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
                logging.warning(
                    "Base model %s cannot be continued. Falling back to training from scratch.",
                    lstm_path,
                )
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
