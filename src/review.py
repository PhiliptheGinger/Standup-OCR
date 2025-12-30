"""Interactive review utilities for confirming OCR results."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable, Iterable, Iterator, Optional, Protocol, Sequence, Set

import cv2
import numpy as np
from PIL import Image

from .ocr import ocr_detailed
from .preprocessing import preprocess_image
from .training import SUPPORTED_EXTENSIONS
from .gpt_ocr import GPTTranscriber, GPTTranscriptionError


class ReviewAborted(RuntimeError):
    """Raised when the operator aborts the review session."""


@dataclass
class ReviewConfig:
    """Configuration options for :class:`ReviewSession`."""

    threshold: float = 70.0
    model_path: Optional[Path] = None
    tessdata_dir: Optional[Path] = None
    psm: int = 6
    train_dir: Path = Path("train")
    preview: bool = True
    full_image_gpt: bool = True


@dataclass
class ReviewPromptContext:
    image_path: Path
    bbox: "ReviewSession.BoundingBox"
    snippet: np.ndarray
    recognised: str
    tesseract_guess: str
    gpt_guess: Optional[str]
    full_image_guess: Optional[str]
    confidence: float
    preview: Optional[Path]


class ReviewPromptHandler(Protocol):
    def __call__(self, context: ReviewPromptContext) -> Optional[str]:
        ...


class ReviewSession:
    """Manage the lifecycle of an interactive review session."""

    def __init__(
        self,
        config: ReviewConfig,
        *,
        log_path: Optional[Path] = None,
        on_sample_saved: Optional[Callable[[Path, str, Path], None]] = None,
        transcriber: Optional[GPTTranscriber] = None,
        gpt_max_images: Optional[int] = None,
        prompt_handler: Optional["ReviewPromptHandler"] = None,
    ) -> None:
        self.config = config
        self.train_dir = Path(config.train_dir)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path or (self.train_dir / "review_log.jsonl")
        self._processed_keys: Set[str] = set()
        self._load_log()
        self.saved_samples = 0
        self._on_sample_saved = on_sample_saved
        self._transcriber = transcriber
        if gpt_max_images is not None and gpt_max_images < 0:
            raise ValueError("gpt_max_images must be zero or a positive integer")
        self._gpt_max_images = gpt_max_images
        self._gpt_transcriptions = 0
        self._gpt_limit_logged = False
        self._prompt_handler = prompt_handler

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def review_paths(self, paths: Iterable[Path]) -> int:
        """Review every path yielded by ``paths`` and return saved sample count."""

        total = 0
        for path in paths:
            if path.is_dir():
                total += self.review_paths(self._iter_images(path))
                continue
            try:
                total += self.review_image(path)
            except ReviewAborted:
                logging.info("Review aborted by operator.")
                break
        return total

    def review_image(self, image_path: Path) -> int:
        """Process a single image and return the number of saved samples."""

        logging.info("Reviewing %s", image_path)
        processed = preprocess_image(image_path)
        detailed = ocr_detailed(
            image_path,
            model_path=self.config.model_path,
            tessdata_dir=self.config.tessdata_dir,
            psm=self.config.psm,
        )
        if detailed.empty:
            logging.info("No OCR tokens produced for %s", image_path)
            return 0

        candidates = detailed.copy()
        candidates["confidence"] = candidates["confidence"].fillna(-1).astype(float)
        candidates["text"] = candidates["text"].fillna("").astype(str).str.strip()
        candidates = candidates[candidates["confidence"] < self.config.threshold]
        candidates = candidates[candidates["text"].ne("")]
        if candidates.empty:
            logging.info(
                "All tokens in %s meet the confidence threshold %.1f",
                image_path,
                self.config.threshold,
            )
            return 0

        saved = 0
        image_height, image_width = processed.shape[:2]
        full_image_tokens = self._prepare_full_image_suggestions(
            image_path, processed, detailed
        )
        full_image_index = 0

        def consume_full_image_suggestion() -> Optional[str]:
            nonlocal full_image_index
            if not full_image_tokens:
                return None
            while full_image_index < len(full_image_tokens):
                suggestion = full_image_tokens[full_image_index].strip()
                full_image_index += 1
                if suggestion:
                    return suggestion
            return None

        for _, row in candidates.iterrows():
            bbox = self._extract_bbox(row, image_width, image_height)
            if bbox is None:
                continue
            key = self._make_key(image_path, bbox)
            if key in self._processed_keys:
                logging.debug("Skipping previously confirmed snippet %s", key)
                continue

            snippet = processed[bbox.top : bbox.bottom, bbox.left : bbox.right]
            if snippet.size == 0:
                continue

            tesseract_guess = row.get("text", "")
            recognised = tesseract_guess
            full_image_guess = consume_full_image_suggestion()
            gpt_guess = self._maybe_transcribe_with_gpt(snippet, image_path, tesseract_guess)
            if gpt_guess:
                recognised = gpt_guess
            elif full_image_guess and (
                not recognised or len(recognised) <= 2
            ):
                recognised = full_image_guess
            elif not recognised and full_image_guess:
                recognised = full_image_guess

            preview_location: Optional[Path] = None
            if self._prompt_handler is None:
                preview_location = self._preview_snippet(snippet)

            context = ReviewPromptContext(
                image_path=image_path,
                bbox=bbox,
                snippet=snippet,
                recognised=recognised,
                tesseract_guess=tesseract_guess,
                gpt_guess=gpt_guess,
                full_image_guess=full_image_guess,
                confidence=float(row.get("confidence", -1) or -1),
                preview=preview_location,
            )
            corrected = self._prompt(context)
            if corrected is None:
                continue

            snippet_path = self._save_snippet(snippet, image_path, corrected)
            self._append_log(
                key,
                image_path,
                bbox,
                recognised,
                corrected,
                snippet_path,
                tesseract_guess=tesseract_guess,
                gpt_guess=gpt_guess,
                full_image_guess=full_image_guess,
            )
            saved += 1
            self.saved_samples += 1
            logging.info("Saved %s with label '%s'", snippet_path.name, corrected)
            self._notify_sample_saved(image_path, corrected, snippet_path)

        return saved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @dataclass(frozen=True)
    class BoundingBox:
        left: int
        top: int
        right: int
        bottom: int

    def _extract_bbox(
        self, row, max_width: int, max_height: int
    ) -> Optional["ReviewSession.BoundingBox"]:
        try:
            left = int(row.get("left", 0))
            top = int(row.get("top", 0))
        except (TypeError, ValueError):
            return None
        width = row.get("width")
        height = row.get("height")
        right = row.get("right")
        bottom = row.get("bottom")
        if width is not None and (right is None or np.isnan(right)):
            right = left + int(width)
        if height is not None and (bottom is None or np.isnan(bottom)):
            bottom = top + int(height)
        try:
            right = int(right)
            bottom = int(bottom)
        except (TypeError, ValueError):
            return None

        left = max(0, min(left, max_width))
        right = max(0, min(right, max_width))
        top = max(0, min(top, max_height))
        bottom = max(0, min(bottom, max_height))
        if right <= left or bottom <= top:
            return None
        return ReviewSession.BoundingBox(left=left, top=top, right=right, bottom=bottom)

    def _preview_snippet(self, snippet: np.ndarray) -> Optional[Path]:
        if not self.config.preview:
            return None
        image = Image.fromarray(snippet)
        try:
            image.show()
            return None
        except Exception:  # pragma: no cover - depends on local environment
            with NamedTemporaryFile(suffix=".png", delete=False) as handle:
                cv2.imwrite(handle.name, snippet)
                temp_path = Path(handle.name)
            logging.info("Preview saved to %s", temp_path)
            return temp_path

    def _prompt(self, context: ReviewPromptContext) -> Optional[str]:
        if self._prompt_handler is not None:
            return self._prompt_handler(context)
        prompt = self._build_prompt(
            context.image_path,
            context.confidence,
            context.preview,
            context.recognised,
            context.tesseract_guess,
            context.gpt_guess,
            context.full_image_guess,
        )
        return self._prompt_for_text(prompt, context.recognised)

    def _build_prompt(
        self,
        image_path: Path,
        confidence: float,
        preview: Optional[Path],
        recognised: str,
        tesseract_guess: str,
        gpt_guess: Optional[str],
        full_image_guess: Optional[str],
    ) -> str:
        parts = [
            f"Image: {image_path.name}",
            f"Confidence: {confidence if confidence >= 0 else 'n/a'}",
        ]
        if gpt_guess:
            parts.append(f"Suggested (ChatGPT): '{gpt_guess}'")
            if tesseract_guess and tesseract_guess != gpt_guess:
                parts.append(f"Tesseract OCR: '{tesseract_guess}'")
        elif full_image_guess:
            parts.append(f"Suggested (Full image GPT): '{full_image_guess}'")
            if tesseract_guess and tesseract_guess != full_image_guess:
                parts.append(f"Tesseract OCR: '{tesseract_guess}'")
        else:
            parts.append(f"Recognised: '{tesseract_guess}'")
        parts.append("Enter corrected text, [s]kip, or [q]uit.")
        if preview:
            parts.append(f"Preview saved to: {preview}")
        return "\n".join(parts) + "\n> "

    def _prompt_for_text(self, prompt: str, recognised: str) -> Optional[str]:
        while True:
            response = input(prompt).strip()
            if not response:
                if recognised:
                    return recognised
                print("Please enter a value or 's' to skip.")
                continue
            lowered = response.lower()
            if lowered in {"s", "skip"}:
                return None
            if lowered in {"q", "quit"}:
                raise ReviewAborted
            return response

    def _save_snippet(self, snippet: np.ndarray, image_path: Path, label: str) -> Path:
        safe_label = self._slugify(label)
        prefix = self._slugify(image_path.stem) or "snippet"
        base_name = f"{prefix}_{safe_label}"
        counter = 1
        while True:
            suffix = "" if counter == 1 else f"_{counter}"
            candidate = self.train_dir / f"{base_name}{suffix}.png"
            if not candidate.exists():
                break
            counter += 1
        cv2.imwrite(str(candidate), snippet)
        return candidate

    def _slugify(self, value: str) -> str:
        value = value.strip().replace(" ", "-")
        cleaned = [c if c.isalnum() or c in {"-"} else "-" for c in value]
        slug = "".join(cleaned).strip("-")
        return slug or "sample"

    def _make_key(self, image_path: Path, bbox: "ReviewSession.BoundingBox") -> str:
        return f"{image_path.resolve()}:{bbox.left}:{bbox.top}:{bbox.right}:{bbox.bottom}"

    def _load_log(self) -> None:
        if not self.log_path.exists():
            return
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = payload.get("key")
            if isinstance(key, str):
                self._processed_keys.add(key)

    def _append_log(
        self,
        key: str,
        image_path: Path,
        bbox: "ReviewSession.BoundingBox",
        recognised: str,
        corrected: str,
        snippet_path: Path,
        *,
        tesseract_guess: Optional[str] = None,
        gpt_guess: Optional[str] = None,
        full_image_guess: Optional[str] = None,
    ) -> None:
        entry = {
            "key": key,
            "image": str(image_path),
            "bbox": {
                "left": bbox.left,
                "top": bbox.top,
                "right": bbox.right,
                "bottom": bbox.bottom,
            },
            "recognised": recognised,
            "corrected": corrected,
            "snippet": str(snippet_path),
        }
        if tesseract_guess is not None:
            entry["tesseract"] = tesseract_guess
        if gpt_guess is not None:
            entry["gpt_guess"] = gpt_guess
        if full_image_guess is not None:
            entry["full_image_guess"] = full_image_guess
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._processed_keys.add(key)

    def _notify_sample_saved(self, image_path: Path, label: str, snippet_path: Path) -> None:
        if self._on_sample_saved is None:
            return
        try:
            self._on_sample_saved(image_path, label, snippet_path)
        except Exception:  # pragma: no cover - callbacks are user supplied
            logging.exception("Sample saved callback failed")

    def _iter_images(self, folder: Path) -> Iterator[Path]:
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path

    def _maybe_transcribe_with_gpt(
        self,
        snippet: np.ndarray,
        image_path: Path,
        tesseract_guess: str,
    ) -> Optional[str]:
        if not self._should_use_gpt():
            return None

        temp_path: Optional[Path] = None
        try:
            with NamedTemporaryFile(suffix=".png", delete=False) as handle:
                success = cv2.imwrite(handle.name, snippet)
                temp_path = Path(handle.name)
            if not success or temp_path is None:
                return None
            hint = f"Tesseract OCR guess: {tesseract_guess}" if tesseract_guess else None
            text = self._transcriber.transcribe(temp_path, hint_text=hint, use_cache=False)
        except GPTTranscriptionError as exc:
            logging.error(
                "ChatGPT OCR failed for snippet from %s: %s",
                image_path.name,
                exc,
            )
            return None
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError:
                    pass

        text = text.strip()
        if not text:
            return None

        self._gpt_transcriptions += 1
        logging.info(
            "ChatGPT OCR suggestion for %s: %s",
            image_path.name,
            text,
        )
        return text

    def _prepare_full_image_suggestions(
        self,
        image_path: Path,
        processed: np.ndarray,
        detailed,
    ) -> Optional[Sequence[str]]:
        if not self.config.full_image_gpt:
            return None
        if self._transcriber is None:
            logging.debug(
                "Full-image GPT transcription skipped for %s because GPT OCR is disabled.",
                image_path,
            )
            return None
        hint_text = self._build_full_image_hint(detailed)
        text = self._transcribe_full_image(processed, image_path, hint_text)
        if not text:
            return None
        tokens = [token for token in re.split(r"\s+", text) if token]
        if not tokens:
            return None
        return tokens

    def _build_full_image_hint(self, detailed) -> Optional[str]:
        if detailed is None or detailed.empty:
            return None
        texts = [
            str(value).strip()
            for value in detailed.get("text", [])
            if isinstance(value, str) and value.strip()
        ]
        if not texts:
            return None
        joined = " ".join(texts)
        return f"Tesseract OCR full text: {joined}"

    def _transcribe_full_image(
        self,
        image: np.ndarray,
        image_path: Path,
        hint_text: Optional[str],
    ) -> Optional[str]:
        if not self._should_use_gpt():
            return None

        temp_path: Optional[Path] = None
        try:
            with NamedTemporaryFile(suffix=".png", delete=False) as handle:
                success = cv2.imwrite(handle.name, image)
                temp_path = Path(handle.name)
            if not success or temp_path is None:
                return None
            text = self._transcriber.transcribe(
                temp_path,
                hint_text=hint_text,
                use_cache=False,
            )
        except GPTTranscriptionError as exc:
            logging.error(
                "ChatGPT OCR failed for %s during full-image pass: %s",
                image_path.name,
                exc,
            )
            return None
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError:
                    pass

        text = text.strip()
        if not text:
            return None

        self._gpt_transcriptions += 1
        logging.info(
            "ChatGPT OCR full-image suggestion for %s: %s",
            image_path.name,
            text,
        )
        return text

    def _should_use_gpt(self) -> bool:
        if self._transcriber is None:
            return False
        if self._gpt_max_images is None:
            return True
        if self._gpt_max_images <= 0:
            return False
        if self._gpt_transcriptions < self._gpt_max_images:
            return True
        if not self._gpt_limit_logged:
            logging.info(
                "ChatGPT OCR limit of %d snippet(s) reached; falling back to Tesseract suggestions.",
                self._gpt_max_images,
            )
            self._gpt_limit_logged = True
        return False


__all__ = [
    "ReviewConfig",
    "ReviewSession",
    "ReviewAborted",
    "ReviewPromptContext",
    "ReviewPromptHandler",
]
