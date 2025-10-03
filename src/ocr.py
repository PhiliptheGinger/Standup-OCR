"""OCR utilities built on top of pytesseract."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
import pytesseract

from .preprocessing import preprocess_image

PathLike = Union[str, Path]


def _array_to_pil(array: np.ndarray) -> Image.Image:
    """Convert a numpy array to a PIL Image."""
    if array.ndim == 2:
        mode = "L"
    else:
        mode = "RGB"
    return Image.fromarray(array, mode=mode)


def ocr_image(
    image_path: PathLike,
    *,
    model_path: Optional[PathLike] = None,
    tessdata_dir: Optional[PathLike] = None,
    psm: int = 6,
) -> str:
    """Run OCR on ``image_path`` using an optional custom Tesseract model.

    Parameters
    ----------
    image_path:
        The image to recognise.
    model_path:
        Optional path to a ``.traineddata`` model created by :func:`train_model`.
        The file should live inside the ``models/`` directory by default.
    tessdata_dir:
        Override the tessdata directory that contains the trained model files.
        If left as ``None`` the directory that holds ``model_path`` will be used.
    psm:
        Page segmentation mode passed to Tesseract. The default (6) works for
        uniform lines of text.

    Returns
    -------
    str
        The recognised text.
    """

    image_path = Path(image_path)
    logging.info("Running OCR on %s", image_path)

    processed = preprocess_image(image_path)
    pil_image = _array_to_pil(processed)

    config_parts = [f"--psm {psm}"]

    lang = "eng"
    if model_path:
        model_path = Path(model_path)
        lang = model_path.stem
        if tessdata_dir is None:
            tessdata_dir = model_path.parent
        tessdata_dir = Path(tessdata_dir)
        config_parts.append(f'--tessdata-dir "{tessdata_dir}"')
        logging.debug("Using custom model %s (lang=%s)", model_path, lang)

    config = " ".join(config_parts)

    text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
    logging.debug("Recognised text: %s", text.strip())
    return text.strip()
