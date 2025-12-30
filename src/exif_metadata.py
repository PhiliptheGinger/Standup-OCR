"""Helpers for encoding Standup-OCR metadata in EXIF tags."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ExifTags


def _find_tag(name: str) -> int:
    for tag, label in ExifTags.TAGS.items():
        if label == name:
            return tag
    raise KeyError(f"EXIF tag '{name}' is not available in this Pillow build.")


_IMAGE_DESCRIPTION_TAG = _find_tag("ImageDescription")
_METADATA_SIGNATURE = "standup-ocr"


def encode_metadata(image: Image.Image, payload: Dict[str, Any]) -> bytes:
    """Return EXIF bytes embedding ``payload`` using the project signature."""

    container = {
        "signature": _METADATA_SIGNATURE,
        "version": 1,
        "payload": payload,
    }
    exif = image.getexif()
    exif[_IMAGE_DESCRIPTION_TAG] = json.dumps(container, ensure_ascii=False)
    return exif.tobytes()


def decode_metadata(source: Path | Image.Image) -> Optional[Dict[str, Any]]:
    """Read the embedded metadata from ``source`` if present."""

    if isinstance(source, Image.Image):
        exif = source.getexif()
    else:
        with Image.open(source) as image:
            exif = image.getexif()
    raw = exif.get(_IMAGE_DESCRIPTION_TAG)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("signature") != _METADATA_SIGNATURE:
        return None
    payload = data.get("payload")
    if isinstance(payload, dict):
        return payload
    return None


__all__ = ["encode_metadata", "decode_metadata"]
