"""Foreground heuristics for filtering noisy Kraken line crops."""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:  # pragma: no cover - permit flat imports in tests
    from .gpt_ocr import GPTTranscriber, GPTTranscriptionError
except ImportError:  # pragma: no cover
    from gpt_ocr import GPTTranscriber, GPTTranscriptionError  # type: ignore

log = logging.getLogger(__name__)

DEFAULT_GPT_FILTER_MODEL = "gpt-4o-mini"
DEFAULT_GPT_FILTER_PROMPT = (
    "You will be shown a small cropped image from a handwriting dataset. "
    "Reply with KEEP if it contains any handwriting, lettering, or printed text. "
    "Reply with DROP if it only contains blank paper, background, or noise. "
    "Respond with a single word: KEEP or DROP."
)


@dataclass
class ForegroundFilterConfig:
    """Thresholds that describe the minimum foreground signal."""

    ink_ratio_threshold: float = 0.01
    edge_density_threshold: float = 0.005
    contrast_threshold: float = 5.0
    ink_margin: float = 0.002
    edge_margin: float = 0.002
    contrast_margin: float = 1.0


def _ensure_grayscale(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        gray = np.array(image.convert("L"), dtype=np.uint8)
        return gray

    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype("uint8", copy=False)

    if array.ndim == 3 and array.shape[2] in (3, 4):
        code = cv2.COLOR_BGRA2GRAY if array.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(array, code)

    raise ValueError("Unsupported image data for foreground measurement")


def compute_foreground_stats(image: Image.Image | np.ndarray) -> Dict[str, float]:
    """Return ink ratio, edge density, and contrast metrics for ``image``."""

    gray = _ensure_grayscale(image)
    total_pixels = float(gray.size)
    if total_pixels == 0:
        return {"ink_ratio": 0.0, "edge_density": 0.0, "contrast": 0.0}

    contrast = float(gray.std())
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    ink_pixels = float(np.count_nonzero(binary == 0))
    ink_ratio = ink_pixels / total_pixels

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / total_pixels

    return {
        "ink_ratio": ink_ratio,
        "edge_density": edge_density,
        "contrast": contrast,
    }


def analyze_foreground(
    stats: Dict[str, float],
    config: ForegroundFilterConfig,
) -> Tuple[bool, bool]:
    """Return (keep, borderline) for the supplied measurements."""

    ink = stats.get("ink_ratio", 0.0)
    edge = stats.get("edge_density", 0.0)
    contrast = stats.get("contrast", 0.0)

    keep = (
        ink >= config.ink_ratio_threshold
        and edge >= config.edge_density_threshold
        and contrast >= config.contrast_threshold
    )

    borderline = False
    if not keep:
        borderline = (
            ink >= max(0.0, config.ink_ratio_threshold - config.ink_margin)
            or edge >= max(0.0, config.edge_density_threshold - config.edge_margin)
            or contrast >= max(0.0, config.contrast_threshold - config.contrast_margin)
        )

    return keep, borderline


class GPTForegroundVerifier:
    """Optional second-stage classifier that asks ChatGPT about a crop."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_GPT_FILTER_MODEL,
        prompt: str = DEFAULT_GPT_FILTER_PROMPT,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.cache_dir = cache_dir
        self._transcriber: Optional[GPTTranscriber] = None
        self._disabled = False

    def _ensure_transcriber(self) -> Optional[GPTTranscriber]:
        if self._disabled:
            return None
        if self._transcriber is not None:
            return self._transcriber
        try:
            self._transcriber = GPTTranscriber(
                model=self.model,
                prompt=self.prompt,
                cache_dir=self.cache_dir,
                max_output_tokens=16,
            )
        except GPTTranscriptionError as exc:
            log.warning(
                "Disabling GPT foreground filtering because the client could not be initialised: %s",
                exc,
            )
            self._disabled = True
            return None
        return self._transcriber

    def decide(self, image: Image.Image) -> Optional[bool]:
        if self._disabled:
            return None
        transcriber = self._ensure_transcriber()
        if transcriber is None:
            return None

        fd, tmp_name = tempfile.mkstemp(prefix="foreground-filter-", suffix=".png")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            image.save(tmp_path)
            response = transcriber.generate(
                tmp_path,
                temperature=0,
                max_output_tokens=8,
            )
            decision = (response.output_text or "").strip().upper()
            if decision.startswith("KEEP"):
                return True
            if decision.startswith("DROP"):
                return False
            log.debug("GPT foreground filter returned unexpected text: %s", decision)
            return None
        except GPTTranscriptionError as exc:
            log.warning("GPT foreground filter failed, disabling future GPT checks: %s", exc)
            self._disabled = True
            return None
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
