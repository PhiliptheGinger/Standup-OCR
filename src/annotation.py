"""GUI for manually annotating handwriting samples."""
from __future__ import annotations

import csv
import logging
import re
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk
import pytesseract
from pytesseract import Output


TokenOrder = Tuple[int, int, int, int, int]
LineKey = Tuple[int, int, int]


@dataclass
class OverlayBox:
    """Track an editable overlay rectangle and its associated widgets."""

    rect_id: int
    entry: tk.Entry
    window_id: int
    token: OcrToken
    is_manual: bool = False
    selected: bool = False


CONTROL_MASK = 0x0004
SHIFT_MASK = 0x0001


def _prepare_image(image: Image.Image) -> Image.Image:
    """Return an image with EXIF orientation applied."""

    return ImageOps.exif_transpose(image)


@dataclass
class AnnotationItem:
    """Represent a single image queued for annotation."""

    path: Path
    label: Optional[str] = None
    status: Optional[str] = None
    saved_path: Optional[Path] = None


@dataclass
class OcrToken:
    """Representation of an OCR token used for overlay editing."""

    text: str
    bbox: Tuple[int, int, int, int]
    order_key: TokenOrder
    line_key: LineKey


def prepare_image(path: Path) -> Image.Image:
    """Open ``path`` and apply EXIF-based orientation for consistent display."""

    with Image.open(path) as src:
        prepared = ImageOps.exif_transpose(src)
        return prepared.copy()


class AnnotationApp:
    """Tkinter-based interface for stepping through a set of images."""

    MAX_SIZE = (900, 700)

    def __init__(
        self,
        master: tk.Tk,
        items: Iterable[AnnotationItem],
        train_dir: Path,
        log_path: Optional[Path] = None,
        on_sample_saved: Optional[Callable[[Path], None]] = None,
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
        self._on_sample_saved = on_sample_saved

        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.canvas_image_id: Optional[int] = None
        self.overlay_entries: list[tk.Entry] = []
        self.overlay_items: list[OverlayBox] = []
        self.rect_to_overlay: Dict[int, OverlayBox] = {}
        self.selected_rects: Set[int] = set()
        self.current_tokens: list[OcrToken] = []
        self.display_scale: Tuple[float, float] = (1.0, 1.0)
        self.manual_token_counter = 0
        self._drag_start: Optional[Tuple[float, float]] = None
        self._active_temp_rect: Optional[int] = None
        self._marquee_rect: Optional[int] = None
        self._modifier_drag = False
        self._pressed_overlay: Optional[OverlayBox] = None

        self.filename_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="select")
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
        button_sequences = (
            "<ButtonPress-1>",
            "<Shift-ButtonPress-1>",
            "<Control-ButtonPress-1>",
            "<Control-Shift-ButtonPress-1>",
        )
        for sequence in button_sequences:
            self.canvas.bind(sequence, self._on_canvas_button_press)
        motion_sequences = (
            "<B1-Motion>",
            "<Shift-B1-Motion>",
            "<Control-B1-Motion>",
            "<Control-Shift-B1-Motion>",
        )
        for sequence in motion_sequences:
            self.canvas.bind(sequence, self._on_canvas_drag)
        release_sequences = (
            "<ButtonRelease-1>",
            "<Shift-ButtonRelease-1>",
            "<Control-ButtonRelease-1>",
            "<Control-Shift-ButtonRelease-1>",
        )
        for sequence in release_sequences:
            self.canvas.bind(sequence, self._on_canvas_release)

        toolbar = tk.Frame(container)
        toolbar.pack(anchor="w", pady=(0, 8))

        select_btn = tk.Radiobutton(toolbar, text="Select", variable=self.mode_var, value="select")
        select_btn.pack(side="left")
        draw_btn = tk.Radiobutton(toolbar, text="Draw", variable=self.mode_var, value="draw")
        draw_btn.pack(side="left", padx=(8, 0))

        delete_btn = tk.Button(toolbar, text="Delete Selected", command=self._delete_selected, state=tk.DISABLED)
        delete_btn.pack(side="left", padx=(16, 0))
        self.delete_button = delete_btn

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

        back_btn = tk.Button(buttons, text="Back", command=self.back)
        back_btn.pack(side="left", padx=4)
        self.back_button = back_btn
        self.back_button.config(state=tk.DISABLED)

        confirm_btn = tk.Button(buttons, text="Confirm", command=self.confirm, default=tk.ACTIVE)
        confirm_btn.pack(side="left", padx=4)

        skip_btn = tk.Button(buttons, text="Skip", command=self.skip)
        skip_btn.pack(side="left", padx=4)

        unsure_btn = tk.Button(buttons, text="Unsure", command=self.unsure)
        unsure_btn.pack(side="left", padx=4)

        self.status_label = tk.Label(container, textvariable=self.status_var, fg="gray")
        self.status_label.pack(anchor="w")

        self.master.bind("<Escape>", self._on_exit)
        self.master.bind("<Alt-Left>", self._on_back)
        self.master.bind("<Delete>", self._on_delete_key)
        self.master.bind("<BackSpace>", self._on_delete_key)
        self.master.protocol("WM_DELETE_WINDOW", self._on_exit)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_confirm(self, event: Optional[tk.Event]) -> None:
        self.confirm()

    def _on_back(self, event: Optional[tk.Event]) -> None:
        self.back()

    def _on_exit(self, event: Optional[tk.Event] = None) -> None:
        if messagebox.askokcancel("Quit", "Abort annotation and close the window?"):
            self.master.destroy()

    def _on_delete_key(self, event: Optional[tk.Event]) -> str:
        self._delete_selected()
        return "break"

    def confirm(self) -> None:
        label = self._get_transcription_text()
        if not label:
            messagebox.showinfo("Missing text", "Enter a transcription or choose Skip/Unsure.")
            return

        item = self.items[self.index]
        try:
            saved_path = self._save_annotation(item.path, label)
        except OSError as exc:
            messagebox.showerror("Save failed", f"Could not save annotation: {exc}")
            return
        item.label = label
        item.status = "confirmed"
        item.saved_path = saved_path
        self._append_log(item.path, label, "confirmed", saved_path)
        self.status_var.set(f"Saved to {saved_path.name}")
        callback = getattr(self, "_on_sample_saved", None)
        if callback is not None and saved_path is not None:
            callback(saved_path)
        self._advance()

    def skip(self) -> None:
        item = self.items[self.index]
        item.label = ""
        item.status = "skipped"
        item.saved_path = None
        self._append_log(item.path, item.label, "skipped", None)
        self.status_var.set("Skipped")
        self._advance()

    def unsure(self) -> None:
        item = self.items[self.index]
        label = self._get_transcription_text()
        item.label = label
        item.status = "unsure"
        item.saved_path = None
        self._append_log(item.path, label, "unsure", None)
        self.status_var.set("Marked as unsure")
        self._advance()

    def back(self) -> None:
        if self.index == 0:
            if hasattr(self, "back_button"):
                self.back_button.config(state=tk.DISABLED)
            self.status_var.set("Already at the first item.")
            return
        self.index -= 1
        self._show_current(revisit=True)

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

    def _show_current(self, *, revisit: bool = False) -> None:
        item = self.items[self.index]
        self.filename_var.set(f"{item.path.name} ({self.index + 1}/{len(self.items)})")
        self._user_modified_transcription = False
        if hasattr(self, "back_button"):
            state = tk.NORMAL if self.index > 0 else tk.DISABLED
            self.back_button.config(state=state)
        self._display_item(item)
        self.entry_widget.focus_set()
        if revisit:
            reminder = "Returned to previous item; previous response has not been re-recorded."
            previous_status = ""
            if hasattr(self.status_var, "get"):
                previous_status = self.status_var.get() or ""
            if reminder not in previous_status:
                message = f"{previous_status} {reminder}".strip()
                self.status_var.set(message)

    def _display_item(self, item: AnnotationItem) -> None:
        path = item.path
        try:
            with Image.open(path) as image:
                image = _prepare_image(image)
                image = image.convert("RGBA")
                image.thumbnail(self.MAX_SIZE, Image.LANCZOS)
                prepared_image = image.copy()
        except Exception as exc:  # pragma: no cover - GUI feedback only
            messagebox.showerror("Error", f"Could not open {path.name}: {exc}")
            self.skip()
            return

        tokens: list[OcrToken] = []
        prefilled = False
        if item.status == "confirmed" and item.label:
            self._set_transcription(item.label)
            if item.saved_path:
                saved_name = Path(item.saved_path).name or str(item.saved_path)
                self.status_var.set(f"Previously saved to {saved_name}")
            else:
                self.status_var.set("Previously confirmed.")
            prefilled = True
        elif item.status == "unsure":
            self._set_transcription(item.label or "")
            self.status_var.set("Previously marked as unsure.")
            prefilled = True
        elif item.status == "skipped":
            self._set_transcription("")
            self.status_var.set("Previously skipped.")
            prefilled = True
        elif item.label:
            self._set_transcription(item.label)
            self.status_var.set("Previously annotated.")
            prefilled = True

        if not prefilled:
            tokens = self._extract_tokens(prepared_image)
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

        self._display_image(prepared_image, tokens)
        prepared_image.close()

    def _display_image(self, image: Image.Image, tokens: Sequence[OcrToken]) -> None:
        base_width, base_height = image.size
        display_image = image.copy().convert("RGBA")
        display_image.thumbnail(self.MAX_SIZE, Image.LANCZOS)
        scale_x = display_image.width / base_width if base_width else 1.0
        scale_y = display_image.height / base_height if base_height else 1.0
        self.display_scale = (scale_x, scale_y)
        self.manual_token_counter = 0

        photo = ImageTk.PhotoImage(display_image)
        self.current_photo = photo

        self._clear_overlay_entries()
        self.canvas.delete("all")
        self.overlay_items = []
        self.overlay_entries = []
        self.rect_to_overlay = {}
        self.selected_rects.clear()
        self._refresh_delete_button()
        self.current_tokens = []
        self.canvas_image_id = self.canvas.create_image(0, 0, image=photo, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, display_image.width, display_image.height))
        self.canvas.focus_set()

        for token in tokens:
            if not token.text:
                continue
            left, top, right, bottom = token.bbox
            disp_left = left * scale_x
            disp_top = top * scale_y
            disp_right = right * scale_x
            disp_bottom = bottom * scale_y
            overlay = self._create_overlay_widget(
                token,
                disp_left,
                disp_top,
                disp_right,
                disp_bottom,
                preset_text=token.text,
                is_manual=False,
            )

        if tokens:
            self._update_tokens_snapshot()
            self._update_combined_transcription()

    def _create_overlay_widget(
        self,
        token: OcrToken,
        disp_left: float,
        disp_top: float,
        disp_right: float,
        disp_bottom: float,
        *,
        preset_text: str = "",
        is_manual: bool,
    ) -> OverlayBox:
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

        entry_width = max(4, int(abs(disp_right - disp_left) / 8))
        entry = tk.Entry(self.canvas, width=entry_width)
        if preset_text:
            entry.insert(0, preset_text)
        entry.bind("<KeyRelease>", self._on_overlay_modified)

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

        overlay = OverlayBox(
            rect_id=rect,
            entry=entry,
            window_id=window_id,
            token=token,
            is_manual=is_manual,
        )
        self.overlay_items.append(overlay)
        self.overlay_entries.append(entry)
        self.rect_to_overlay[rect] = overlay
        self._update_tokens_snapshot()
        return overlay

    def _display_to_base_bbox(self, bbox: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        scale_x, scale_y = self.display_scale
        left, top, right, bottom = bbox
        inv_x = 1.0 / scale_x if scale_x else 1.0
        inv_y = 1.0 / scale_y if scale_y else 1.0
        return (
            int(round(left * inv_x)),
            int(round(top * inv_y)),
            int(round(right * inv_x)),
            int(round(bottom * inv_y)),
        )

    def _next_manual_keys(self) -> Tuple[TokenOrder, LineKey]:
        self.manual_token_counter += 1
        index = self.manual_token_counter
        order_key: TokenOrder = (9999, 0, 0, index, index)
        line_key: LineKey = (9999, 0, index)
        return order_key, line_key

    def _create_manual_overlay(self, bbox: Tuple[float, float, float, float]) -> Optional[OverlayBox]:
        left, top, right, bottom = bbox
        if abs(right - left) < 4 or abs(bottom - top) < 4:
            return None
        base_bbox = self._display_to_base_bbox(bbox)
        order_key, line_key = self._next_manual_keys()
        token = OcrToken(text="", bbox=base_bbox, order_key=order_key, line_key=line_key)
        overlay = self._create_overlay_widget(
            token,
            left,
            top,
            right,
            bottom,
            preset_text="",
            is_manual=True,
        )
        overlay.entry.focus_set()
        self._clear_selection()
        self.selected_rects.add(overlay.rect_id)
        self._set_box_selected(overlay, True)
        self._update_entry_focus()
        self._refresh_delete_button()
        return overlay

    def _update_tokens_snapshot(self) -> None:
        self.current_tokens = [item.token for item in self.overlay_items]

    def _set_box_selected(self, overlay: OverlayBox, selected: bool) -> None:
        overlay.selected = selected
        outline = "#F2994A" if selected else "#2F80ED"
        width = 2 if selected else 1
        try:
            self.canvas.itemconfigure(overlay.rect_id, outline=outline, width=width)
        except tk.TclError:
            return

    def _clear_selection(self) -> None:
        for rect in list(self.selected_rects):
            overlay = self.rect_to_overlay.get(rect)
            if overlay is None:
                continue
            self._set_box_selected(overlay, False)
        self.selected_rects.clear()
        self._refresh_delete_button()

    def _refresh_delete_button(self) -> None:
        if hasattr(self, "delete_button"):
            state = tk.NORMAL if self.selected_rects else tk.DISABLED
            try:
                self.delete_button.config(state=state)
            except tk.TclError:
                pass

    def _update_entry_focus(self) -> None:
        if len(self.selected_rects) != 1:
            return
        rect_id = next(iter(self.selected_rects))
        overlay = self.rect_to_overlay.get(rect_id)
        if overlay is None:
            return
        try:
            overlay.entry.focus_set()
        except tk.TclError:
            pass

    def _delete_selected(self) -> None:
        overlays = [self.rect_to_overlay.get(rect) for rect in list(self.selected_rects)]
        overlays = [overlay for overlay in overlays if overlay is not None]
        if not overlays:
            return
        for overlay in overlays:
            self._remove_overlay(overlay)
        self.selected_rects.clear()
        self._refresh_delete_button()
        self._update_tokens_snapshot()
        self._update_combined_transcription()

    def _remove_overlay(self, overlay: OverlayBox) -> None:
        try:
            overlay.entry.destroy()
        except tk.TclError:
            pass
        self.canvas.delete(overlay.rect_id)
        self.canvas.delete(overlay.window_id)
        overlay.selected = False
        self.selected_rects.discard(overlay.rect_id)
        if overlay.rect_id in self.rect_to_overlay:
            del self.rect_to_overlay[overlay.rect_id]
        try:
            self.overlay_items.remove(overlay)
        except ValueError:
            pass
        try:
            self.overlay_entries.remove(overlay.entry)
        except ValueError:
            pass

    def _get_mode(self) -> str:
        if hasattr(self, "mode_var"):
            try:
                return self.mode_var.get()
            except AttributeError:
                return str(self.mode_var)
        return "select"

    def _event_has_ctrl(self, event: tk.Event) -> bool:
        return bool(getattr(event, "state", 0) & CONTROL_MASK)

    def _event_has_shift(self, event: tk.Event) -> bool:
        return bool(getattr(event, "state", 0) & SHIFT_MASK)

    def _find_overlay_at(self, x: float, y: float) -> Optional[OverlayBox]:
        for overlay in reversed(self.overlay_items):
            coords = self.canvas.coords(overlay.rect_id)
            if len(coords) != 4:
                continue
            left, top, right, bottom = coords
            if left <= x <= right and top <= y <= bottom:
                return overlay
        return None

    def _overlays_in_rect(self, bbox: Tuple[float, float, float, float]) -> List[OverlayBox]:
        left, top, right, bottom = bbox
        selected: list[OverlayBox] = []
        for overlay in self.overlay_items:
            coords = self.canvas.coords(overlay.rect_id)
            if len(coords) != 4:
                continue
            o_left, o_top, o_right, o_bottom = coords
            if o_left >= right or o_right <= left or o_top >= bottom or o_bottom <= top:
                continue
            selected.append(overlay)
        return selected

    def _apply_single_selection(self, overlay: OverlayBox, additive: bool) -> None:
        if additive:
            if overlay.rect_id in self.selected_rects:
                self.selected_rects.remove(overlay.rect_id)
                self._set_box_selected(overlay, False)
            else:
                self.selected_rects.add(overlay.rect_id)
                self._set_box_selected(overlay, True)
        else:
            if self.selected_rects == {overlay.rect_id}:
                pass
            else:
                self._clear_selection()
                self.selected_rects.add(overlay.rect_id)
                self._set_box_selected(overlay, True)
        self._refresh_delete_button()
        self._update_entry_focus()

    def _apply_marquee_selection(self, overlays: Sequence[OverlayBox], additive: bool) -> None:
        if not additive:
            self._clear_selection()
        for overlay in overlays:
            self.selected_rects.add(overlay.rect_id)
            self._set_box_selected(overlay, True)
        self._refresh_delete_button()
        self._update_entry_focus()

    def _on_canvas_button_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        self._drag_start = (event.x, event.y)
        self._active_temp_rect = None
        self._marquee_rect = None
        self._modifier_drag = False
        mode = self._get_mode()
        if mode == "draw":
            self._active_temp_rect = self.canvas.create_rectangle(
                event.x,
                event.y,
                event.x,
                event.y,
                outline="#2F80ED",
                dash=(4, 2),
                tags="overlay-temp",
            )
        else:
            has_modifier = self._event_has_ctrl(event) or self._event_has_shift(event)
            self._modifier_drag = has_modifier
            overlay = self._find_overlay_at(event.x, event.y)
            self._pressed_overlay = overlay
            if overlay is not None and not has_modifier:
                self._apply_single_selection(overlay, additive=False)
            elif overlay is None and not has_modifier:
                self._clear_selection()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        mode = self._get_mode()
        if mode == "draw":
            if self._active_temp_rect is None:
                self._active_temp_rect = self.canvas.create_rectangle(
                    self._drag_start[0],
                    self._drag_start[1],
                    event.x,
                    event.y,
                    outline="#2F80ED",
                    dash=(4, 2),
                    tags="overlay-temp",
                )
            else:
                self.canvas.coords(
                    self._active_temp_rect,
                    self._drag_start[0],
                    self._drag_start[1],
                    event.x,
                    event.y,
                )
        else:
            if not self._modifier_drag:
                return
            if self._marquee_rect is None:
                self._marquee_rect = self.canvas.create_rectangle(
                    self._drag_start[0],
                    self._drag_start[1],
                    event.x,
                    event.y,
                    outline="#F2994A",
                    dash=(3, 2),
                    tags="marquee",
                )
            else:
                self.canvas.coords(
                    self._marquee_rect,
                    self._drag_start[0],
                    self._drag_start[1],
                    event.x,
                    event.y,
                )

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self._drag_start is None:
            return
        mode = self._get_mode()
        if mode == "draw":
            if self._active_temp_rect is not None:
                coords = self.canvas.coords(self._active_temp_rect)
                self.canvas.delete(self._active_temp_rect)
                if len(coords) == 4:
                    left, top, right, bottom = coords
                    bbox = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))
                    overlay = self._create_manual_overlay(bbox)
                    if overlay is not None:
                        self._update_combined_transcription()
            self._active_temp_rect = None
        else:
            if self._marquee_rect is not None:
                coords = self.canvas.coords(self._marquee_rect)
                self.canvas.delete(self._marquee_rect)
                if len(coords) == 4:
                    left, top, right, bottom = coords
                    bbox = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))
                    overlays = self._overlays_in_rect(bbox)
                    additive = self._modifier_drag
                    self._apply_marquee_selection(overlays, additive=additive)
            else:
                overlay = self._pressed_overlay
                if overlay is not None:
                    additive = self._event_has_ctrl(event) or self._event_has_shift(event)
                    if additive or overlay.rect_id not in self.selected_rects:
                        self._apply_single_selection(overlay, additive=additive)
                elif not (self._event_has_ctrl(event) or self._event_has_shift(event)):
                    self._clear_selection()
        self._pressed_overlay = None
        self._drag_start = None
        self._marquee_rect = None
        self._modifier_drag = False

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
            self._apply_transcription_to_overlays()

    def _apply_transcription_to_overlays(self) -> None:
        if self._setting_transcription:
            return

        text = self.entry_widget.get("1.0", tk.END)
        tokens = re.findall(r"\S+", text)

        self._setting_transcription = True
        try:
            for index, entry in enumerate(self.overlay_entries):
                value = tokens[index] if index < len(tokens) else ""
                try:
                    entry.delete(0, tk.END)
                    if value:
                        entry.insert(0, value)
                except tk.TclError:
                    continue
        finally:
            self._setting_transcription = False

        previous_flag = self._user_modified_transcription
        self._user_modified_transcription = False
        self._update_combined_transcription()
        self._user_modified_transcription = previous_flag

    def _update_combined_transcription(self) -> None:
        if self._user_modified_transcription:
            return
        text = self._compose_transcription()
        self._set_transcription(text)

    def _compose_transcription(self) -> str:
        lines: dict[LineKey, list[Tuple[int, str]]] = {}
        overlays = sorted(self.overlay_items, key=lambda item: item.token.order_key)
        for overlay in overlays:
            value = overlay.entry.get().strip()
            if not value:
                continue
            token = overlay.token
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
        for overlay in list(getattr(self, "overlay_items", [])):
            try:
                overlay.entry.destroy()
            except tk.TclError:
                pass
        self.overlay_entries = []
        self.overlay_items = []
        self.rect_to_overlay = {}
        self.selected_rects.clear()
        self._refresh_delete_button()
        self.current_tokens = []

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
        slug = "".join(cleaned)
        slug = re.sub("-+", "-", slug).strip("-")
        if len(slug) > 60:
            slug = slug[:60].rstrip("-")
        return slug or "sample"


def _train_model(*args, **kwargs):
    if __package__:
        from .training import train_model as _impl
    else:  # pragma: no cover - fallback for test imports
        from training import train_model as _impl
    return _impl(*args, **kwargs)


@dataclass
class AnnotationAutoTrainConfig:
    auto_train: int
    output_model: str
    model_dir: Path
    base_lang: str
    max_iterations: int
    tessdata_dir: Optional[Path] = None


class AnnotationTrainer:
    """Schedule background training runs as annotations are confirmed."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        train_dir: Path,
        config: AnnotationAutoTrainConfig,
    ) -> None:
        if config.auto_train <= 0:
            raise ValueError("auto_train must be a positive integer")
        self.master = master
        self.train_dir = Path(train_dir)
        self.config = config
        self._pending: deque[Path] = deque()
        self._seen_samples: List[Path] = []
        self._running = False

    @property
    def seen_samples(self) -> List[Path]:
        return list(self._seen_samples)

    def __call__(self, sample_path: Path) -> None:
        path = Path(sample_path)
        self._pending.append(path)
        self._seen_samples.append(path)
        self.master.after(0, self._maybe_train)

    def _maybe_train(self) -> None:
        if self._running:
            return
        if len(self._pending) < self.config.auto_train:
            return

        batch = [self._pending.popleft() for _ in range(self.config.auto_train)]
        self._running = True

        def worker() -> None:
            try:
                logging.info(
                    "Auto-training triggered by %d new annotations.",
                    len(batch),
                )
                model_path = _train_model(
                    self.train_dir,
                    self.config.output_model,
                    model_dir=self.config.model_dir,
                    tessdata_dir=self.config.tessdata_dir,
                    base_lang=self.config.base_lang,
                    max_iterations=self.config.max_iterations,
                )
                logging.info("Updated model saved to %s", model_path)
            except Exception:  # pragma: no cover - logging side effects
                logging.exception("Auto-training failed.")
            finally:
                self._running = False
                self.master.after(0, self._maybe_train)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()


def annotate_images(
    sources: Iterable[Path],
    train_dir: Path,
    *,
    log_path: Optional[Path] = None,
    auto_train_config: Optional[AnnotationAutoTrainConfig] = None,
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

    callback = None
    if auto_train_config is not None:
        callback = AnnotationTrainer(
            root,
            train_dir=train_dir,
            config=auto_train_config,
        )

    app = AnnotationApp(root, items, train_dir, log_path, on_sample_saved=callback)
    root.mainloop()
