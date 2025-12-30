"""Image preprocessing utilities for handwriting OCR."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image, ExifTags, ImageOps

import logging


PathLike = Union[str, Path]


@dataclass
class OrientationOptions:
    """Configuration for :func:`normalize_page_orientation`."""

    max_skew: float = 25.0
    force_landscape: bool = True
    force_upright: bool = True


@dataclass(init=False)
class OrientationResult:
    """Metadata describing how an image was normalized."""

    applied: bool = False
    angle: float = 0.0
    rotated_quadrants: int = 0
    flipped: bool = False
    original_width: int = 0
    original_height: int = 0
    normalized_width: int = 0
    normalized_height: int = 0
    
    def __init__(
        self,
        *,
        applied: bool = False,
        angle: float = 0.0,
        rotated_quadrants: int = 0,
        flipped: bool = False,
        original_width: int = 0,
        original_height: int = 0,
        normalized_width: int = 0,
        normalized_height: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        if width is not None:
            original_width = width
        if height is not None:
            original_height = height
        self.applied = applied
        self.angle = angle
        self.rotated_quadrants = rotated_quadrants
        self.flipped = flipped
        self.original_width = original_width
        self.original_height = original_height
        self.normalized_width = normalized_width or original_width
        self.normalized_height = normalized_height or original_height
    
    @property
    def width(self) -> int:
        """Backward compatibility: return original width."""
        return self.original_width
    
    @property
    def height(self) -> int:
        """Backward compatibility: return original height."""
        return self.original_height


def denormalize_coordinates(
    coords: list[tuple[float, float]],
    original_size: tuple[int, int],
    normalized_size: tuple[int, int],
    rotation_meta: OrientationResult,
) -> list[tuple[float, float]]:
    """Inverse the rotations/flips applied during normalization to map back to original image space.
    
    Args:
        coords: Points in normalized (rotated) image space.
        original_size: (width, height) of the original image before normalization.
        normalized_size: (width, height) of the normalized image.
        rotation_meta: OrientationResult describing the transformations applied.
    
    Returns:
        Points mapped back to original image space.
    """
    if not rotation_meta.applied or not coords:
        return coords
    
    orig_w, orig_h = original_size
    result = list(coords)
    
    # Work backwards through the transformations that were applied:
    # 1. force_upright flip (180°) - quads 2
    # 2. force_landscape rotation (90° CCW) - quads 1
    
    # Reverse force_upright flip first (if applied and rotated_quadrants >= 2)
    if rotation_meta.flipped and rotation_meta.rotated_quadrants >= 2:
        # After landscape rotation, dimensions are swapped
        # If landscape was applied: normalized is rotated, so we need to flip in rotated space
        # A 180° flip: (x, y) -> (W-1-x, H-1-y)
        norm_w, norm_h = normalized_size
        result = [(norm_w - 1.0 - x, norm_h - 1.0 - y) for x, y in result]
    
    # Reverse force_landscape rotation (90° CCW) by rotating 90° CW 3 times (or 1 time CW)
    # 90° CCW: (x, y) in HxW -> (y, W-1-x) in WxH (width and height swapped)
    # 90° CW (reverse): (x, y) in WxH -> (H-1-y, x) in HxW
    if rotation_meta.rotated_quadrants >= 1 and not rotation_meta.flipped:
        # Only reverse the rotation part if not flipped
        # After CCW rotation, dimensions became swapped
        norm_w, norm_h = normalized_size
        # Reverse 90° CCW by rotating 90° CW: (x, y) -> (H-1-y, x) but in rotated coordinates
        result = [(norm_h - 1.0 - y, x) for x, y in result]
    elif rotation_meta.rotated_quadrants >= 1 and rotation_meta.flipped:
        # If we had both rotation and flip, reverse rotation after unreversing flip
        norm_w, norm_h = normalized_size
        result = [(norm_h - 1.0 - y, x) for x, y in result]
    
    # Clamp to original bounds
    result = [
        (max(0.0, min(float(orig_w - 1), x)), max(0.0, min(float(orig_h - 1), y)))
        for x, y in result
    ]
    return result


def _estimate_skew_angle(gray: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    inverted = cv2.bitwise_not(thresh)
    coords = cv2.findNonZero(inverted)
    if coords is None or len(coords) < 100:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    return angle


def _rotate_image(array: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return array
    height, width = array.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    rotated = cv2.warpAffine(
        array,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _should_flip_vertical(gray: np.ndarray) -> bool:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    height = gray.shape[0]
    if height < 4:
        return False
    ink = 255 - gray
    midpoint = height // 2
    top = float(np.sum(ink[:midpoint]))
    bottom = float(np.sum(ink[midpoint:]))
    if bottom == 0:
        return False
    return top > bottom * 1.15


def normalize_page_orientation(
    image: Image.Image,
    *,
    options: OrientationOptions | None = None,
) -> tuple[Image.Image, OrientationResult]:
    """Return ``image`` with deskew/orientation normalization applied."""

    opts = options or OrientationOptions()
    prepared = ImageOps.exif_transpose(image)
    array = np.array(prepared)
    result = OrientationResult(
        applied=False,
        original_width=array.shape[1],
        original_height=array.shape[0],
        normalized_width=array.shape[1],
        normalized_height=array.shape[0],
    )

    if array.ndim == 2:
        gray = array
    else:
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    angle = _estimate_skew_angle(gray)
    if opts.max_skew > 0 and abs(angle) <= opts.max_skew and abs(angle) >= 0.3:
        array = _rotate_image(array, angle)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        result.angle = angle
        result.applied = True

    if opts.force_landscape and array.shape[0] > array.shape[1] * 1.1:
        array = np.rot90(array, 1)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        result.rotated_quadrants = (result.rotated_quadrants + 1) % 4
        result.applied = True

    if opts.force_upright and _should_flip_vertical(gray):
        array = np.rot90(array, 2)
        result.rotated_quadrants = (result.rotated_quadrants + 2) % 4
        result.flipped = True
        result.applied = True

    result.normalized_width = array.shape[1]
    result.normalized_height = array.shape[0]
    normalized = Image.fromarray(array)
    return normalized, result


def load_normalized_image(
    image_path: PathLike, *, options: OrientationOptions | None = None
) -> tuple[Image.Image, OrientationResult]:
    """Load ``image_path`` and apply :func:`normalize_page_orientation`."""

    image_path = Path(image_path)
    with Image.open(image_path) as src:
        normalized, meta = normalize_page_orientation(src, options=options)
        return normalized.copy(), meta


def preprocess_image(
    image_path: PathLike,
    *,
    resize_width: int = 1800,
    adaptive: bool = True,
    force_landscape: bool = False,
) -> np.ndarray:
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
    force_landscape:
        When ``True`` rotate portrait images counterclockwise so that the output
        is landscape oriented. Portrait images remain upright by default unless
        EXIF metadata explicitly requests a rotation.

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

    # Attempt to orient the image according to any available EXIF metadata.
    try:
        with Image.open(image_path) as pil_img:
            orientation_tag = next(
                (
                    tag
                    for tag, name in ExifTags.TAGS.items()
                    if name == "Orientation"
                ),
                None,
            )
            if orientation_tag is not None:
                exif = pil_img._getexif()
                if exif is not None:
                    orientation_value = exif.get(orientation_tag, 1)
                    if orientation_value == 3:
                        image = cv2.rotate(image, cv2.ROTATE_180)
                    elif orientation_value == 6:
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    elif orientation_value == 8:
                        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        # If EXIF data is unavailable or unreadable, proceed without rotating.
        pass

    # Optionally normalize orientation so that images are landscape.
    if force_landscape and image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

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
