"""GUI for manually annotating handwriting samples."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk
import pytesseract
from pytesseract import Output


def _prepare_image(image: Image.Image) -> Image.Image:
    """Return an image with EXIF orientation applied."""

    return ImageOps.exif_transpose(image)


@dataclass
class AnnotationItem:
    """Represent a single image queued for annotation."""

    path: Path


@dataclass
class OcrToken:
    """Representation of an OCR token used for overlay editing."""

    text: str
    bbox: Tuple[int, int, int, int]
    order_key: TokenOrder
    line_key: LineKey


@dataclass
class OverlayItem:
    """UI artefacts representing an editable OCR token."""

    token: OcrToken
    rect_id: int
    window_id: int
    entry: tk.Entry
    selected: bool = False
    fade_job: Optional[str] = None


@dataclass
class RemovedOverlay:
    """Snapshot of an overlay used to support undo behaviour."""

    token: OcrToken
    text: str
    index: int


def prepare_image(path: Path) -> Image.Image:
    """Open ``path`` and apply EXIF-based orientation for consistent display."""

    with Image.open(path) as src:
        prepared = ImageOps.exif_transpose(src)
        return prepared.copy()


class AnnotationApp:
    """Tkinter-based interface for stepping through a set of images."""

    MAX_SIZE = (900, 700)
    FADE_DELAY_MS = 150

    def __init__(
        self,
        master: tk.Tk,
        items: Iterable[AnnotationItem],
        train_dir: Path,
        log_path: Optional[Path] = None,
    ) -> None:
        self.master = master
        self.items: List[AnnotationItem] = list(items)
        if not self.items:
            raise ValueError("No images were provided for annotation.")
        self.index = 0
        self.train_dir = Path(train_dir)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        if self.log_path is not None:
            self.log_path = Path(self.log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.canvas_image_id: Optional[int] = None
        self.overlay_items: list[OverlayItem] = []
        self.current_tokens: list[OcrToken] = []
        self._undo_stack: list[list[RemovedOverlay]] = []
        self._scale_x = 1.0
        self._scale_y = 1.0

        self.filename_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self._user_modified_transcription = False
        self._setting_transcription = False

        self._build_ui()
        self._show_current()

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.master.title("Standup-OCR Annotation")
        self.master.geometry("1024x840")

        container = tk.Frame(self.master, padx=12, pady=12)
        container.pack(fill="both", expand=True)

        header = tk.Label(container, textvariable=self.filename_var, font=("TkDefaultFont", 14, "bold"))
        header.pack(anchor="w")

        self.canvas = tk.Canvas(container, bd=1, relief="sunken", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, pady=12)

        entry_frame = tk.Frame(container)
        entry_frame.pack(fill="x", pady=(0, 8))

        tk.Label(entry_frame, text="Transcription:").pack(side="left")
        text_widget = tk.Text(entry_frame, height=4, wrap="word")
        text_widget.pack(side="left", fill="both", expand=True, padx=(8, 0))
        text_widget.bind("<Control-Return>", self._on_confirm)
        text_widget.bind("<Command-Return>", self._on_confirm)
        text_widget.bind("<Key>", self._on_transcription_modified)
        self.entry_widget = text_widget

        buttons = tk.Frame(container)
        buttons.pack(pady=(0, 8))

        confirm_btn = tk.Button(buttons, text="Confirm", command=self.confirm, default=tk.ACTIVE)
        confirm_btn.pack(side="left", padx=4)

        skip_btn = tk.Button(buttons, text="Skip", command=self.skip)
        skip_btn.pack(side="left", padx=4)

        unsure_btn = tk.Button(buttons, text="Unsure", command=self.unsure)
        unsure_btn.pack(side="left", padx=4)

        self.status_label = tk.Label(container, textvariable=self.status_var, fg="gray")
        self.status_label.pack(anchor="w")

        self.master.bind("<Escape>", self._on_exit)
        self.master.bind("<Delete>", self._on_delete_selected)
        self.master.bind("<BackSpace>", self._on_delete_selected)
        self.master.bind("<Control-z>", self._on_undo)
        self.master.bind("<Control-Z>", self._on_undo)
        self.master.protocol("WM_DELETE_WINDOW", self._on_exit)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_confirm(self, event: Optional[tk.Event]) -> None:
        self.confirm()

    def _on_exit(self, event: Optional[tk.Event] = None) -> None:
        if messagebox.askokcancel("Quit", "Abort annotation and close the window?"):
            self.master.destroy()

    def confirm(self) -> None:
        label = self._get_transcription_text()
        if not label:
            messagebox.showinfo("Missing text", "Enter a transcription or choose Skip/Unsure.")
            return

        item = self.items[self.index]
        saved_path = self._save_annotation(item.path, label)
        self._append_log(item.path, label, "confirmed", saved_path)
        self.status_var.set(f"Saved to {saved_path.name}")
        self._advance()

    def skip(self) -> None:
        item = self.items[self.index]
        self._append_log(item.path, "", "skipped", None)
        self.status_var.set("Skipped")
        self._advance()

    def unsure(self) -> None:
        item = self.items[self.index]
        self._append_log(item.path, self._get_transcription_text(), "unsure", None)
        self.status_var.set("Marked as unsure")
        self._advance()

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _advance(self) -> None:
        self.index += 1
        if self.index >= len(self.items):
            messagebox.showinfo("Complete", "All images have been processed.")
            self.master.destroy()
            return
        self._show_current()

    def _show_current(self) -> None:
        item = self.items[self.index]
        self.filename_var.set(f"{item.path.name} ({self.index + 1}/{len(self.items)})")
        self._user_modified_transcription = False
        self._display_item(item.path)
        self.entry_widget.focus_set()

    def _display_item(self, path: Path) -> None:
        try:
            with Image.open(path) as image:
                image = _prepare_image(image)
                image = image.convert("RGBA")
                image.thumbnail(self.MAX_SIZE, Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
        except Exception as exc:  # pragma: no cover - GUI feedback only
            messagebox.showerror("Error", f"Could not open {path.name}: {exc}")
            self.skip()
            return

        tokens = self._extract_tokens(image)
        suggestion = self._compose_text_from_tokens(tokens)
        if suggestion:
            self._set_transcription(suggestion)
            self.status_var.set("Pre-filled transcription using OCR result.")
        else:
            self._set_transcription("")
            filename_hint = self._suggest_label(path)
            if filename_hint:
                self.status_var.set(
                    "OCR produced no suggestion; using filename hint: "
                    f"{filename_hint}"
                )
            else:
                self.status_var.set("OCR produced no suggestion; please transcribe manually.")

        self._display_image(image, tokens)
        image.close()

    def _display_image(self, image: Image.Image, tokens: Sequence[OcrToken]) -> None:
        base_width, base_height = image.size
        display_image = image.copy().convert("RGBA")
        display_image.thumbnail(self.MAX_SIZE, Image.LANCZOS)
        self._scale_x = display_image.width / base_width if base_width else 1.0
        self._scale_y = display_image.height / base_height if base_height else 1.0

        photo = ImageTk.PhotoImage(display_image)
        self.current_photo = photo

        self._clear_overlay_entries()
        if hasattr(self, "_undo_stack"):
            self._undo_stack.clear()
        else:  # pragma: no cover - defensive for partially initialised instances
            self._undo_stack = []
        self.canvas.delete("all")
        self.current_tokens = list(tokens)
        self.canvas_image_id = self.canvas.create_image(0, 0, image=photo, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, display_image.width, display_image.height))

        for token in tokens:
            if not token.text:
                continue
            self._add_overlay_item(token, token.text)

        if tokens:
            self._update_combined_transcription()

    def _add_overlay_item(self, token: OcrToken, text: str, index: Optional[int] = None) -> None:
        left, top, right, bottom = token.bbox
        disp_left = left * self._scale_x
        disp_top = top * self._scale_y
        disp_right = right * self._scale_x
        disp_bottom = bottom * self._scale_y

        rect = self.canvas.create_rectangle(
            disp_left,
            disp_top,
            disp_right,
            disp_bottom,
            outline="#2F80ED",
            width=1,
            tags="overlay",
        )
        self.canvas.tag_raise(rect)

        entry_width = max(4, int((disp_right - disp_left) / 8))
        entry = tk.Entry(self.canvas, width=entry_width)
        if text:
            entry.insert(0, text)
        entry.bind("<KeyRelease>", self._on_overlay_modified)
        entry.bind("<Button-1>", lambda event, item_token=token: self._focus_overlay_by_token(item_token))
        entry.bind("<FocusIn>", lambda event, token=token: self._focus_overlay_by_token(token))
        entry.bind("<Enter>", lambda event, token=token: self._on_overlay_enter(token))
        entry.bind("<Leave>", lambda event, token=token: self._on_overlay_leave(token))

        desired_top = disp_top - 24
        if desired_top < 0:
            desired_top = disp_top

        window_id = self.canvas.create_window(
            disp_left,
            desired_top,
            anchor="nw",
            window=entry,
            tags="overlay",
        )
        self.canvas.tag_raise(window_id)

        item = OverlayItem(token=token, rect_id=rect, window_id=window_id, entry=entry)
        self.canvas.tag_bind(rect, "<Button-1>", lambda event, token=token: self._on_overlay_click(event, token))
        self.canvas.tag_bind(rect, "<Enter>", lambda event, token=token: self._on_overlay_enter(token))
        self.canvas.tag_bind(rect, "<Leave>", lambda event, token=token: self._on_overlay_leave(token))

        if index is None or index >= len(self.overlay_items):
            self.overlay_items.append(item)
            self.current_tokens.append(token)
        else:
            self.overlay_items.insert(index, item)
            self.current_tokens.insert(index, token)

    def _focus_overlay_by_token(self, token: OcrToken) -> None:
        item = self._find_overlay(token)
        if item is None:
            return
        self._select_overlay(item, additive=False)
        self._set_entry_visibility(item, True)

    def _find_overlay(self, token: OcrToken) -> Optional[OverlayItem]:
        for item in self.overlay_items:
            if item.token is token:
                return item
        return None

    def _on_overlay_click(self, event: tk.Event, token: OcrToken) -> None:
        item = self._find_overlay(token)
        if item is None:
            return
        ctrl = bool(event.state & 0x0004)
        shift = bool(event.state & 0x0001)
        additive = ctrl or shift
        if additive and item.selected:
            self._set_overlay_selected(item, False)
        else:
            self._select_overlay(item, additive=additive)
        self._set_entry_visibility(item, True)
        item.entry.focus_set()

    def _select_overlay(self, target: OverlayItem, additive: bool = False) -> None:
        if not additive:
            for item in self.overlay_items:
                if item.selected:
                    self._set_overlay_selected(item, False)
        self._set_overlay_selected(target, True)

    def _set_overlay_selected(self, item: OverlayItem, value: bool) -> None:
        item.selected = value
        outline = "#F2994A" if value else "#2F80ED"
        width = 2 if value else 1
        try:
            self.canvas.itemconfigure(item.rect_id, outline=outline, width=width)
        except tk.TclError:
            pass
        if value:
            self._set_entry_visibility(item, True)

    def _on_delete_selected(self, event: Optional[tk.Event]) -> str:
        if self._delete_selected_overlays():
            return "break"
        return ""

    def _delete_selected_overlays(self) -> bool:
        removed: list[RemovedOverlay] = []
        for index in reversed(range(len(self.overlay_items))):
            item = self.overlay_items[index]
            if not item.selected:
                continue
            removed.append(RemovedOverlay(token=item.token, text=item.entry.get(), index=index))
            if item.fade_job is not None:
                try:
                    self.master.after_cancel(item.fade_job)
                except tk.TclError:
                    pass
                item.fade_job = None
            try:
                self.canvas.delete(item.rect_id)
                self.canvas.delete(item.window_id)
            except tk.TclError:
                pass
            try:
                item.entry.destroy()
            except tk.TclError:
                pass
            del self.overlay_items[index]
            del self.current_tokens[index]

        if not removed:
            return False

        # Maintain original order for undo restoration.
        removed.sort(key=lambda overlay: overlay.index)
        self._undo_stack.append(removed)
        self._update_combined_transcription()
        return True

    def _on_undo(self, event: Optional[tk.Event]) -> str:
        if not self._undo_stack:
            return ""
        overlays = self._undo_stack.pop()
        for overlay in overlays:
            self._add_overlay_item(overlay.token, overlay.text, index=overlay.index)
        self._update_combined_transcription()
        return "break"

    def _on_overlay_enter(self, token: OcrToken) -> None:
        item = self._find_overlay(token)
        if item is None or item.selected:
            return
        if item.fade_job is not None:
            try:
                self.master.after_cancel(item.fade_job)
            except tk.TclError:
                pass
        item.fade_job = self.master.after(
            self.FADE_DELAY_MS, lambda item=item: self._set_entry_visibility(item, False)
        )

    def _on_overlay_leave(self, token: OcrToken) -> None:
        item = self._find_overlay(token)
        if item is None:
            return
        if item.fade_job is not None:
            try:
                self.master.after_cancel(item.fade_job)
            except tk.TclError:
                pass
            item.fade_job = None
        self._set_entry_visibility(item, True)

    def _set_entry_visibility(self, item: OverlayItem, visible: bool) -> None:
        try:
            state = "normal" if visible else "hidden"
            self.canvas.itemconfigure(item.window_id, state=state)
        except tk.TclError:
            pass
        if visible and item.fade_job is not None:
            item.fade_job = None

    def _extract_tokens(self, image: Image.Image) -> List[OcrToken]:
        ocr_image: Optional[Image.Image] = None
        try:
            ocr_image = image.copy()
            if ocr_image.mode not in {"RGB", "L"}:
                converted = ocr_image.convert("RGB")
                ocr_image.close()
                ocr_image = converted
            data = pytesseract.image_to_data(
                ocr_image,
                config="--psm 6",
                output_type=Output.DICT,
            )
        except pytesseract.TesseractNotFoundError as exc:
            logging.warning("Tesseract not found: %s", exc)
            return []
        except pytesseract.TesseractError as exc:
            logging.warning("Tesseract error: %s", exc)
            return []
        finally:
            if ocr_image is not None:
                ocr_image.close()

        if not data or "text" not in data:
            return []

        tokens: List[OcrToken] = []
        length = len(data.get("text", []))
        for index in range(length):
            text = (data["text"][index] or "").strip()
            if not text:
                continue
            try:
                left = int(data.get("left", [0])[index])
                top = int(data.get("top", [0])[index])
                width = int(data.get("width", [0])[index])
                height = int(data.get("height", [0])[index])
                page = int(data.get("page_num", [1])[index])
                block = int(data.get("block_num", [0])[index])
                paragraph = int(data.get("par_num", [0])[index])
                line = int(data.get("line_num", [0])[index])
                word = int(data.get("word_num", [index + 1])[index]) or (index + 1)
            except (TypeError, ValueError):
                continue

            bbox = (left, top, left + width, top + height)
            order_key: TokenOrder = (page, block, paragraph, line, word)
            line_key: LineKey = (page, block, line)
            tokens.append(OcrToken(text=text, bbox=bbox, order_key=order_key, line_key=line_key))

        tokens.sort(key=lambda token: token.order_key)
        return tokens

    def _compose_text_from_tokens(self, tokens: Sequence[OcrToken]) -> str:
        if not tokens:
            return ""

        lines: dict[LineKey, list[Tuple[int, str]]] = {}
        for token in tokens:
            lines.setdefault(token.line_key, []).append((token.order_key[-1], token.text))

        composed: list[str] = []
        for line_key in sorted(lines.keys()):
            words = [word for _, word in sorted(lines[line_key], key=lambda item: item[0])]
            composed.append(" ".join(words))
        return "\n".join(composed)

    def _on_overlay_modified(self, event: tk.Event | None) -> None:
        self._user_modified_transcription = False
        self._update_combined_transcription()

    def _on_transcription_modified(self, event: tk.Event | None) -> None:
        if not self._setting_transcription:
            self._user_modified_transcription = True

    def _update_combined_transcription(self) -> None:
        if self._user_modified_transcription:
            return
        text = self._compose_transcription()
        self._set_transcription(text)

    def _compose_transcription(self) -> str:
        lines: dict[LineKey, list[Tuple[int, str]]] = {}
        for item in self.overlay_items:
            value = item.entry.get().strip()
            if not value:
                continue
            token = item.token
            lines.setdefault(token.line_key, []).append((token.order_key[-1], value))

        composed: list[str] = []
        for line_key in sorted(lines.keys()):
            words = [word for _, word in sorted(lines[line_key], key=lambda item: item[0])]
            composed.append(" ".join(words))
        return "\n".join(composed)

    def _set_transcription(self, value: str) -> None:
        self._setting_transcription = True
        self.entry_widget.delete("1.0", tk.END)
        if value:
            self.entry_widget.insert("1.0", value)
        self._setting_transcription = False
        self._user_modified_transcription = False

    def _get_transcription_text(self) -> str:
        return self.entry_widget.get("1.0", tk.END).strip()

    def _clear_overlay_entries(self) -> None:
        for item in self.overlay_items:
            if item.fade_job is not None:
                try:
                    self.master.after_cancel(item.fade_job)
                except tk.TclError:
                    pass
                item.fade_job = None
            try:
                self.canvas.delete(item.rect_id)
                self.canvas.delete(item.window_id)
            except tk.TclError:
                pass
            try:
                item.entry.destroy()
            except tk.TclError:
                pass
        self.overlay_items.clear()

    def _suggest_label(self, path: Path) -> str:
        stem = path.stem
        parts = stem.split("_", 1)
        if len(parts) == 2 and parts[1]:
            candidate = parts[1]
        else:
            candidate = parts[0]
        candidate = candidate.replace("-", " ")
        return candidate.strip()

    def _save_annotation(self, path: Path, label: str) -> Path:
        safe_label = self._slugify(label)
        prefix = self._slugify(path.stem)
        base_name = f"{prefix}_{safe_label}" if safe_label else prefix
        counter = 1
        while True:
            suffix = "" if counter == 1 else f"_{counter}"
            candidate = self.train_dir / f"{base_name}{suffix}.png"
            if not candidate.exists():
                break
            counter += 1

        with Image.open(path) as image:
            prepared_image = _prepare_image(image)
            if prepared_image.mode not in {"RGB", "L"}:
                prepared_image = prepared_image.convert("RGB")
            prepared_image.save(candidate)
        return candidate

    def _append_log(self, source: Path, label: str, status: str, saved_path: Optional[Path]) -> None:
        if self.log_path is None:
            return
        exists = self.log_path.exists()
        with self.log_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "status", "label", "saved_path"])
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "image": str(source),
                    "status": status,
                    "label": label,
                    "saved_path": str(saved_path) if saved_path else "",
                }
            )

    def _slugify(self, value: str) -> str:
        cleaned = [c if c.isalnum() else "-" for c in value.strip().lower()]
        slug = "".join(cleaned).strip("-")
        return slug or "sample"


def annotate_images(
    sources: Iterable[Path],
    train_dir: Path,
    *,
    log_path: Optional[Path] = None,
) -> None:
    """Launch the annotation UI for the provided image paths."""

    items = [AnnotationItem(Path(path)) for path in sources]
    if not items:
        raise ValueError("No images found to annotate.")

    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Tkinter could not be initialised. Ensure a display is available or "
            "use a system package such as python3-tk."
        ) from exc

    app = AnnotationApp(root, items, train_dir, log_path)
    root.mainloop()
