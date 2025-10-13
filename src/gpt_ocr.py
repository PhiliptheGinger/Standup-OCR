"""Utilities for transcribing handwriting samples using ChatGPT's vision API."""
from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

def _load_dotenv(env_path: Path = Path(".env")) -> None:
    """Populate :mod:`os.environ` with variables declared in ``env_path``.

    The implementation is intentionally lightweight so that users can drop a
    ``.env`` file next to the project without needing an additional
    dependency such as :mod:`python-dotenv`. Only ``KEY=VALUE`` assignments are
    supported and existing environment variables always take precedence.
    """

    try:
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip('"').strip("'")
    except OSError:  # pragma: no cover - filesystem edge cases
        # Silently ignore issues loading the .env file; environment variables
        # remain untouched so callers can still configure credentials manually.
        return


_load_dotenv()


try:  # pragma: no cover - import guard for optional dependency
    from openai import OpenAI, OpenAIError
except ImportError as exc:  # pragma: no cover - handled at runtime
    OpenAI = None  # type: ignore[assignment]
    OpenAIError = Exception  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial
    _OPENAI_IMPORT_ERROR = None


DEFAULT_PROMPT = (
    "Transcribe the handwritten text in this image. "
    "Return only the transcription without additional commentary or punctuation fixes."
)


class GPTTranscriptionError(RuntimeError):
    """Raised when an image cannot be transcribed via ChatGPT."""


@dataclass
class GPTTranscriber:
    """Thin wrapper around the OpenAI Responses API for handwriting OCR."""

    model: str = "gpt-4o-mini"
    prompt: str = DEFAULT_PROMPT
    max_output_tokens: int = 256
    cache_dir: Optional[Path] = None
    _client: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if OpenAI is None:  # type: ignore[truthy-bool]
            raise GPTTranscriptionError(
                "The 'openai' package is required for ChatGPT OCR. Install it with 'pip install openai'."
            ) from _OPENAI_IMPORT_ERROR
        try:
            self._client = OpenAI()  # type: ignore[operator]
        except OpenAIError as exc:  # pragma: no cover - network failure
            raise GPTTranscriptionError(
                "Failed to initialise the OpenAI client. Ensure OPENAI_API_KEY is set."
            ) from exc

        if self.cache_dir is not None:
            cache_path = Path(self.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self.cache_dir = cache_path

    def generate(
        self,
        image_path: Path,
        *,
        prompt: Optional[str] = None,
        hint_text: Optional[str] = None,
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ):
        """Execute a multimodal completion request for ``image_path``."""

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        encoded, mime_type = self._encode_image(image_path)

        prompt_text = self.prompt if prompt is None else prompt
        content = []
        if prompt_text:
            content.append({"type": "input_text", "text": prompt_text})
        if hint_text:
            content.append({"type": "input_text", "text": hint_text})
        content.append({
            "type": "input_image",
            "image_url": f"data:{mime_type};base64,{encoded}",
        })

        request: dict = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "max_output_tokens": max_output_tokens or self.max_output_tokens,
        }
        if response_format is not None:
            request["response_format"] = response_format
        if temperature is not None:
            request["temperature"] = temperature

        try:
            return self._client.responses.create(**request)
        except OpenAIError as exc:  # pragma: no cover - network failure
            raise GPTTranscriptionError(f"OpenAI request failed: {exc}") from exc

    def transcribe(self, image_path: Path) -> str:
        """Return the transcription produced by ChatGPT for ``image_path``."""

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.cache_dir is not None:
            cached = self._cache_path(image_path)
            if cached.exists():
                return cached.read_text(encoding="utf-8").strip()

        response = self.generate(image_path)
        text = response.output_text.strip()
        if not text:
            raise GPTTranscriptionError("Received an empty transcription from ChatGPT.")

        if self.cache_dir is not None:
            cached = self._cache_path(image_path)
            cached.write_text(text, encoding="utf-8")

        return text

    def _encode_image(self, image_path: Path) -> Tuple[str, str]:
        data = image_path.read_bytes()
        encoded = base64.b64encode(data).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            mime_type = "image/png"
        return encoded, mime_type

    def _cache_path(self, image_path: Path) -> Path:
        if self.cache_dir is None:  # pragma: no cover - defensive only
            raise RuntimeError("Cache directory was not initialised")
        return self.cache_dir / f"{image_path.stem}.txt"
