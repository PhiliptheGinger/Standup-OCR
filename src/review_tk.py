"""Tkinter-based prompt handler for the review workflow."""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageOps, ImageTk

from .review import ReviewAborted, ReviewPromptContext


@dataclass
class _PendingAction:
    kind: str
    text: Optional[str] = None


class TkReviewPrompt:
    """Interactive Tk prompt handler plugged into :class:`ReviewSession`."""

    def __init__(
        self,
        *,
        page_max_size: tuple[int, int] = (520, 360),
        snippet_max_width: int = 420,
    ) -> None:
        self._page_max_size = page_max_size
        self._snippet_max_width = snippet_max_width
        self._root = tk.Tk()
        self._root.title("Standup-OCR Review")
        self._root.protocol("WM_DELETE_WINDOW", self._on_quit)

        self._status_var = tk.StringVar(value="")
        self._suggestion_var = tk.StringVar(value="")

        self._snippet_canvas = tk.Canvas(self._root, width=snippet_max_width, height=200, bg="#111")
        self._page_canvas = tk.Canvas(self._root, width=page_max_size[0], height=page_max_size[1], bg="#222")

        self._entry = tk.Entry(self._root)
        self._entry.bind("<Return>", lambda _e: self._confirm())
        self._entry.bind("<Control-Return>", lambda _e: self._confirm())

        confirm_btn = tk.Button(self._root, text="Confirm", command=self._confirm)
        skip_btn = tk.Button(self._root, text="Skip", command=self._skip)
        quit_btn = tk.Button(self._root, text="Quit", command=self._on_quit)

        layout = tk.Frame(self._root, padx=12, pady=12)
        layout.pack(fill="both", expand=True)

        canvases = tk.Frame(layout)
        canvases.pack(fill="x", expand=True)
        self._snippet_canvas.pack(side="left", padx=(0, 12))
        self._page_canvas.pack(side="left")

        info = tk.Label(layout, textvariable=self._status_var, anchor="w")
        info.pack(fill="x", pady=(12, 4))

        suggestion = tk.Label(layout, textvariable=self._suggestion_var, anchor="w", fg="#555")
        suggestion.pack(fill="x")

        self._entry.pack(fill="x", pady=(12, 4))

        buttons = tk.Frame(layout)
        buttons.pack(fill="x")
        confirm_btn.pack(side="left")
        skip_btn.pack(side="left", padx=8)
        quit_btn.pack(side="left")

        self._snippet_photo: Optional[ImageTk.PhotoImage] = None
        self._page_photo: Optional[ImageTk.PhotoImage] = None
        self._pending: Optional[_PendingAction] = None

    def prompt(self, context: ReviewPromptContext) -> Optional[str]:
        self._apply_context(context)
        self._pending = None
        self._entry.focus_set()
        while self._pending is None:
            self._root.update_idletasks()
            self._root.update()
        if self._pending.kind == "quit":
            raise ReviewAborted
        if self._pending.kind == "skip":
            return None
        text = self._pending.text or ""
        text = text.strip()
        return text or context.recognised

    def destroy(self) -> None:
        try:
            self._root.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _apply_context(self, context: ReviewPromptContext) -> None:
        self._entry.delete(0, tk.END)
        self._entry.insert(0, context.recognised)
        self._status_var.set(
            f"{context.image_path.name}  |  Confidence: {context.confidence if context.confidence >= 0 else 'n/a'}"
        )
        self._suggestion_var.set(self._format_suggestions(context))
        self._render_snippet(context.snippet)
        self._render_page(context.image_path, context.bbox)

    def _format_suggestions(self, context: ReviewPromptContext) -> str:
        parts = []
        if context.tesseract_guess:
            parts.append(f"Tesseract: {context.tesseract_guess}")
        if context.gpt_guess:
            parts.append(f"GPT: {context.gpt_guess}")
        if context.full_image_guess:
            parts.append(f"Full image: {context.full_image_guess}")
        return " | ".join(parts) if parts else ""

    def _render_snippet(self, snippet) -> None:
        snippet_img = Image.fromarray(snippet)
        width, height = snippet_img.size
        if width == 0 or height == 0:
            return
        scale = min(self._snippet_max_width / width, 3.0)
        if scale < 1.0:
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            snippet_img = snippet_img.resize(new_size, Image.LANCZOS)
        self._snippet_photo = ImageTk.PhotoImage(snippet_img)
        self._snippet_canvas.configure(width=self._snippet_photo.width(), height=self._snippet_photo.height())
        self._snippet_canvas.delete("all")
        self._snippet_canvas.create_image(0, 0, image=self._snippet_photo, anchor="nw")

    def _render_page(self, image_path: Path, bbox) -> None:
        try:
            with Image.open(image_path) as page:
                # Apply EXIF orientation
                page = ImageOps.exif_transpose(page)
                page = page.convert("RGB")
                scale = self._thumbnail_scale(page.size)
                if scale != 1.0:
                    new_size = (max(1, int(page.width * scale)), max(1, int(page.height * scale)))
                    page = page.resize(new_size, Image.LANCZOS)
                draw = ImageDraw.Draw(page)
                rect = (
                    bbox.left * scale,
                    bbox.top * scale,
                    bbox.right * scale,
                    bbox.bottom * scale,
                )
                draw.rectangle(rect, outline="#f97316", width=3)
        except Exception:
            return
        self._page_photo = ImageTk.PhotoImage(page)
        self._page_canvas.configure(width=self._page_photo.width(), height=self._page_photo.height())
        self._page_canvas.delete("all")
        self._page_canvas.create_image(0, 0, image=self._page_photo, anchor="nw")

    def _thumbnail_scale(self, size: tuple[int, int]) -> float:
        width, height = size
        max_w, max_h = self._page_max_size
        scale = min(max_w / width if width else 1.0, max_h / height if height else 1.0, 1.0)
        return scale if scale > 0 else 1.0

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _confirm(self) -> None:
        self._pending = _PendingAction("confirm", text=self._entry.get())

    def _skip(self) -> None:
        self._pending = _PendingAction("skip")

    def _on_quit(self) -> None:
        self._pending = _PendingAction("quit")


__all__ = ["TkReviewPrompt"]