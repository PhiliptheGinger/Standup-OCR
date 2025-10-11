"""OCR utilities built on top of pytesseract.

The module exposes :func:`ocr_detailed`, which yields per-token metadata such
as confidences and bounding boxes, and :func:`ocr_image`, a convenience wrapper
that collapses those tokens into plain text.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from pytesseract import Output

from ..preprocessing import preprocess_image

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

    detailed = ocr_detailed(
        image_path,
        model_path=model_path,
        tessdata_dir=tessdata_dir,
        psm=psm,
    )

    if detailed.empty:
        return ""

    tokens = detailed.copy()
    tokens["text"] = tokens["text"].fillna("").astype(str).str.strip()
    tokens = tokens[tokens["text"].ne("")]
    if tokens.empty:
        return ""

    sort_columns = ["page_num", "block_num", "par_num", "line_num", "word_num"]
    tokens = tokens.sort_values(sort_columns)
    grouped = tokens.groupby(sort_columns[:-1], sort=True)["text"].apply(
        lambda words: " ".join(words)
    )
    text = "\n".join(grouped.tolist())
    logging.debug("Recognised text: %s", text.strip())
    return text.strip()


def ocr_detailed(
    image_path: PathLike,
    *,
    model_path: Optional[PathLike] = None,
    tessdata_dir: Optional[PathLike] = None,
    psm: int = 6,
) -> pd.DataFrame:
    """Run OCR on ``image_path`` and return detailed token metadata.

    The helper centralises Tesseract configuration so both structured metadata
    and plain-text recognition share the same setup.

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
    pandas.DataFrame
        A dataframe containing per-token text, confidence, and bounding boxes.
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
    data = pytesseract.image_to_data(
        pil_image, lang=lang, config=config, output_type=Output.DICT
    )

    detailed = pd.DataFrame(data)
    if detailed.empty:
        return detailed

    detailed = detailed[detailed.get("level") == 5].copy()
    if detailed.empty:
        return detailed

    detailed.rename(columns={"conf": "confidence"}, inplace=True)
    detailed["confidence"] = pd.to_numeric(detailed["confidence"], errors="coerce")
    for column in ["left", "top", "width", "height", "page_num", "block_num", "par_num", "line_num", "word_num"]:
        if column in detailed.columns:
            detailed[column] = pd.to_numeric(detailed[column], errors="coerce")

    if {"left", "width"}.issubset(detailed.columns):
        detailed["right"] = detailed["left"] + detailed["width"]
    if {"top", "height"}.issubset(detailed.columns):
        detailed["bottom"] = detailed["top"] + detailed["height"]

    return detailed.reset_index(drop=True)
