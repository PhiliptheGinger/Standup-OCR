"""Tests for the annotation helpers."""

import sys
from pathlib import Path

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from annotation import _prepare_image


def test_prepare_image_applies_exif_orientation():
    """Images should be rotated according to their EXIF orientation."""

    image = Image.new("RGB", (10, 20), "red")
    exif = image.getexif()
    exif[274] = 6  # Orientation tag: rotate 270 degrees
    image.info["exif"] = exif.tobytes()

    prepared = _prepare_image(image)

    assert prepared.size == (20, 10)
