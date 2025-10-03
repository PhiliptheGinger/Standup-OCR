"""Image preprocessing utilities for handwriting OCR."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

import logging


PathLike = Union[str, Path]


def preprocess_image(image_path: PathLike, *, resize_width: int = 1800, adaptive: bool = True) -> np.ndarray:
    """Load and preprocess an image so it is ready for OCR.

    Parameters
    ----------
    image_path:
        Path to the image that should be cleaned up. Place your handwriting
        samples in the ``train/`` directory before running training. Each file
        name should encode the ground-truth label, e.g. ``a_01.png`` or
        ``word_hello.png``. The part after the first underscore is treated as
        the text value during automated label extraction.
    resize_width:
        When the input is narrower than this value, the image will be scaled up
        while preserving aspect ratio. Larger images are kept as-is. Upscaling
        generally helps Tesseract read small handwriting samples.
    adaptive:
        If ``True`` (default) use adaptive thresholding, otherwise fall back to
        Otsu's global threshold.

    Returns
    -------
    numpy.ndarray
        A processed, single-channel (grayscale) image array that can be fed
        directly to Tesseract or saved to disk.
    """

    image_path = Path(image_path)
    logging.debug("Preprocessing image %s", image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Slight blur removes sensor noise and scanning artifacts.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if adaptive:
        processed = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )
    else:
        _, processed = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Upscale smaller samples to improve recognition of fine handwriting.
    if resize_width and processed.shape[1] < resize_width:
        scale = resize_width / processed.shape[1]
        new_size = (
            int(processed.shape[1] * scale),
            int(processed.shape[0] * scale),
        )
        processed = cv2.resize(processed, new_size, interpolation=cv2.INTER_CUBIC)
        logging.debug(
            "Resized image %s to %s for better OCR readability",
            image_path.name,
            new_size,
        )

    return processed
