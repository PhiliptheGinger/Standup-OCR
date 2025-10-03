from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import pandas as pd

if "cv2" not in sys.modules:  # pragma: no cover - shim when OpenCV is unavailable.
    sys.modules["cv2"] = types.SimpleNamespace()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.ocr as ocr  # noqa: E402  (import after path adjustment)


def _dummy_data_dict():
    return {
        "level": [5, 5],
        "page_num": [1, 1],
        "block_num": [1, 1],
        "par_num": [1, 1],
        "line_num": [1, 1],
        "word_num": [1, 2],
        "left": [10, 50],
        "top": [20, 20],
        "width": [30, 40],
        "height": [10, 12],
        "conf": ["95", "85"],
        "text": ["Hello", "world"],
    }


def test_ocr_detailed_returns_confidence(tmp_path):
    fake_array = np.zeros((10, 10), dtype=np.uint8)

    with (
        patch.object(ocr, "preprocess_image", return_value=fake_array) as preprocess_mock,
        patch.object(ocr.pytesseract, "image_to_data", return_value=_dummy_data_dict()) as data_mock,
    ):
        detailed = ocr.ocr_detailed(tmp_path / "image.png")

    preprocess_mock.assert_called_once()
    data_mock.assert_called_once()

    assert not detailed.empty
    assert "confidence" in detailed
    assert detailed.loc[0, "confidence"] == 95
    assert {"left", "top", "width", "height"}.issubset(detailed.columns)


def test_ocr_image_reassembles_text():
    dataframe = pd.DataFrame(
        {
            "page_num": [1, 1],
            "block_num": [1, 1],
            "par_num": [1, 1],
            "line_num": [1, 2],
            "word_num": [1, 1],
            "text": ["Hello", "world"],
            "confidence": [90.0, 80.0],
            "left": [0, 0],
            "top": [0, 10],
            "width": [10, 10],
            "height": [5, 5],
        }
    )

    with patch.object(ocr, "ocr_detailed", return_value=dataframe) as detailed_mock:
        text = ocr.ocr_image("dummy.png", psm=4)

    detailed_mock.assert_called_once_with(
        "dummy.png", model_path=None, tessdata_dir=None, psm=4
    )
    assert text == "Hello\nworld"
