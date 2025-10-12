"""Utilities for fine-tuning a Tesseract LSTM model."""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2

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


def _discover_images(train_dir: Path) -> List[Path]:
    """Return a sorted list of image paths inside ``train_dir``."""
    images = [
        p
        for p in sorted(train_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(
            f"No training images found in {train_dir}. Place handwriting samples "
            "named like 'word_hello.png' inside this folder."
        )
    return images


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
    gt_file.write_text(label + "\n", encoding="utf-8")

    # Optionally save a preprocessed version alongside the original.
    processed = preprocess_image(image_path)
    processed_path = work_dir / f"{base_name}.png"

    cv2.imwrite(str(processed_path), processed)
    logging.debug("Prepared GT for %s => %s", image_path.name, label)
    return processed_path, gt_file


def _generate_lstmf(processed_image: Path, work_dir: Path) -> Path:
    """Invoke Tesseract to create the .lstmf feature file from an image."""
    base = work_dir / processed_image.stem
    command = [
        "tesseract",
        str(processed_image),
        str(base),
        "--psm",
        "6",
        "--oem",
        "1",
        "nobatch",
        "lstm.train",
    ]
    _run_command(command)
    lstmf_path = base.with_suffix(".lstmf")
    if not lstmf_path.exists():
        raise RuntimeError(f"Tesseract did not produce {lstmf_path}")
    return lstmf_path


def _resolve_tessdata_dir(tessdata_dir: Optional[PathLike]) -> Path:
    if tessdata_dir:
        return Path(tessdata_dir)
    env_dir = os.environ.get("TESSDATA_PREFIX")
    if env_dir:
        return Path(env_dir)
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
    windows_default = Path("C:/Program Files/Tesseract-OCR/tessdata")
    if windows_default.exists():
        return windows_default
    # Default for many linux distributions
    default = Path("/usr/share/tesseract-ocr/4.00/tessdata")
    if default.exists():
        return default
    raise FileNotFoundError(
        "Unable to locate tessdata directory. Set TESSDATA_PREFIX, install Tesseract, "
        "or pass tessdata_dir explicitly (see README for details)."
    )


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
        lstmf_path = _generate_lstmf(processed_path, work_dir)
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
    _run_command(
        [
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
    )

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
