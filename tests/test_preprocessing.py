from pathlib import Path
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cv2 = pytest.importorskip(
    "cv2",
    reason="OpenCV is required to exercise preprocessing orientation",
    exc_type=ImportError,
)

from src.preprocessing import preprocess_image


def _make_portrait_image(path):
    portrait = Image.new("RGB", (100, 200), color="white")
    portrait.save(path)


def test_preprocess_keeps_portrait_orientation_without_exif(tmp_path):
    image_path = tmp_path / "portrait.png"
    _make_portrait_image(image_path)

    processed = preprocess_image(image_path, resize_width=None)

    assert processed.shape[0] > processed.shape[1]


def test_preprocess_can_force_landscape_orientation(tmp_path):
    image_path = tmp_path / "portrait.png"
    _make_portrait_image(image_path)

    processed = preprocess_image(
        image_path, resize_width=None, force_landscape=True
    )

    assert processed.shape[1] > processed.shape[0]
