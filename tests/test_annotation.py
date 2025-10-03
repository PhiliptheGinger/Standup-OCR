from pathlib import Path
from unittest import mock
import sys
import types

from PIL import Image

if "cv2" not in sys.modules:  # pragma: no cover - shim when OpenCV is unavailable.
    sys.modules["cv2"] = types.SimpleNamespace()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation import prepare_image  # noqa: E402  (import after path tweak)


def test_prepare_image_applies_exif_transpose(tmp_path):
    source = tmp_path / "sample.png"
    Image.new("RGB", (10, 20), "white").save(source)

    def fake_transpose(image: Image.Image) -> Image.Image:
        return image.transpose(Image.ROTATE_90)

    with mock.patch("src.annotation.ImageOps.exif_transpose", side_effect=fake_transpose) as transpose:
        result = prepare_image(Path(source))
    try:
        assert result.size == (20, 10)
    finally:
        result.close()

    transpose.assert_called_once()
