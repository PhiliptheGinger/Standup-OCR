"""GUI for manually annotating handwriting samples."""
from __future__ import annotations

import csv
import logging
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk
import pytesseract
from pytesseract import Output


TokenOrder = Tuple[int, int, int, int, int]
LineKey = Tuple[int, int, int]


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


@dataclass
class OverlayItem:
    """UI artefacts representing an editable OCR token."""

    token: OcrToken
    rect_id: int
    window_id: int
    entry: tk.Entry
    is_manual: bool = False
    selected: bool = False
    fade_job: Optional[str] = None
    handles: Dict[str, int] = field(default_factory=dict)


@dataclass
class RemovedOverlay:
    """Snapshot of an overlay used to support undo behaviour."""

    token: OcrToken
    text: str
    index: int
    scale: Tuple[float, float]


@dataclass
class UndoAction:
    """Describe an undoable change that can be replayed."""

    kind: str
    overlays: list[RemovedOverlay] = field(default_factory=list)
    transforms: list[Tuple[OcrToken, Tuple[int, int, int, int]]] = field(default_factory=list)


def prepare_image(path: Path) -> Image.Image:
    """Open ``path`` and apply EXIF-based orientation for consistent display."""

    with Image.open(path) as src:
        prepared = ImageOps.exif_transpose(src)
        return prepared.copy()


class AnnotationApp:
    """Tkinter-based interface for stepping through a set of images."""

    MAX_SIZE = (900, 700)
    FADE_DELAY_MS = 150
    MIN_ZOOM = 0.5
    MAX_ZOOM = 3.0
    ZOOM_STEP = 1.1
    RESIZE_HANDLE = 6
    MIN_DISPLAY_BOX = 8

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
        self.overlay_items: list[OverlayItem] = []
        self.rect_to_overlay: Dict[int, OverlayItem] = {}
        self.selected_rects: Set[int] = set()
        self.handle_to_overlay: Dict[int, Tuple[OverlayItem, str]] = {}
        self._undo_stack: list[UndoAction] = []
        self.current_tokens: list[OcrToken] = []
        self.display_scale: Tuple[float, float] = (1.0, 1.0)
        self._base_display_image: Optional[Image.Image] = None
        self._base_scale: Tuple[float, float] = (1.0, 1.0)
        self.zoom_factor: float = 1.0
        self.manual_token_counter = 0
        self._drag_start: Optional[Tuple[float, float]] = None
        self._active_temp_rect: Optional[int] = None
        self._marquee_rect: Optional[int] = None
        self._marquee_dragging = False
        self._marquee_additive = False
        self._pressed_overlay: Optional[OverlayItem] = None
        self._dragging_overlay: Optional[OverlayItem] = None
        self._dragging_mode: Optional[str] = None
        self._drag_offset: Tuple[float, float] = (0.0, 0.0)
        self._drag_initial_bbox: Optional[Tuple[float, float, float, float]] = None
        self._resize_anchor: Optional[Tuple[float, float]] = None
        self._resize_corner: Optional[str] = None
        self._drag_original_bbox: Optional[Tuple[int, int, int, int]] = None

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

        canvas_frame = tk.Frame(container)
        canvas_frame.pack(fill="both", expand=True, pady=12)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bd=1, relief="sunken", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        v_scroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
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
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind("<Button-4>", self._on_canvas_mousewheel)
        self.canvas.bind("<Button-5>", self._on_canvas_mousewheel)

        toolbar = tk.Frame(container)
        toolbar.pack(anchor="w", pady=(0, 8))

        select_btn = tk.Radiobutton(toolbar, text="Select", variable=self.mode_var, value="select")
        select_btn.pack(side="left")
        draw_btn = tk.Radiobutton(toolbar, text="Draw", variable=self.mode_var, value="draw")
        draw_btn.pack(side="left", padx=(8, 0))
        zoom_btn = tk.Radiobutton(toolbar, text="Zoom", variable=self.mode_var, value="zoom")
        zoom_btn.pack(side="left", padx=(8, 0))

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
        self._base_display_image = display_image.copy()
        self._base_scale = (scale_x, scale_y)
        self.zoom_factor = 1.0
        self.display_scale = self._base_scale
        self.manual_token_counter = 0

        photo = ImageTk.PhotoImage(self._base_display_image)
        self.current_photo = photo

        self._clear_overlay_entries()
        if not hasattr(self, "_undo_stack"):
            self._undo_stack = []
        else:
            self._undo_stack.clear()
        self.canvas.delete("all")
        self.overlay_items = []
        self.overlay_entries = []
        self.rect_to_overlay = {}
        self.selected_rects.clear()
        self._refresh_delete_button()
        self.current_tokens = []
        self.canvas_image_id = self.canvas.create_image(0, 0, image=photo, anchor="nw")
        self.canvas.config(
            scrollregion=(0, 0, self._base_display_image.width, self._base_display_image.height)
        )
        if hasattr(self.canvas, "xview_moveto"):
            self.canvas.xview_moveto(0)
        if hasattr(self.canvas, "yview_moveto"):
            self.canvas.yview_moveto(0)
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
    ) -> OverlayItem:
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

        overlay = OverlayItem(
            token=token,
            rect_id=rect,
            window_id=window_id,
            entry=entry,
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

    def _base_to_display_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        scale_x, scale_y = self.display_scale
        left, top, right, bottom = bbox
        return (
            left * scale_x,
            top * scale_y,
            right * scale_x,
            bottom * scale_y,
        )

    def _get_display_bounds(self) -> Tuple[float, float]:
        if self._base_display_image is None:
            return (0.0, 0.0)
        width, height = self._base_display_image.size
        return (width * self.zoom_factor, height * self.zoom_factor)

    def _update_overlay_display(
        self, overlay: OverlayItem, bbox: Tuple[float, float, float, float]
    ) -> None:
        left, top, right, bottom = bbox
        try:
            self.canvas.coords(overlay.rect_id, left, top, right, bottom)
        except tk.TclError:
            return
        width = max(4, int(max(1, abs(right - left)) / 8))
        try:
            overlay.entry.configure(width=width)
        except tk.TclError:
            pass
        desired_top = top - 24
        if desired_top < 0:
            desired_top = top
        try:
            self.canvas.coords(overlay.window_id, left, desired_top)
        except tk.TclError:
            pass
        self._position_overlay_handles(overlay, (left, top, right, bottom))

    def _determine_resize_corner(
        self, bbox: Tuple[float, float, float, float], x: float, y: float
    ) -> Optional[str]:
        left, top, right, bottom = bbox
        handle = self.RESIZE_HANDLE
        if abs(x - left) <= handle and abs(y - top) <= handle:
            return "nw"
        if abs(x - right) <= handle and abs(y - top) <= handle:
            return "ne"
        if abs(x - left) <= handle and abs(y - bottom) <= handle:
            return "sw"
        if abs(x - right) <= handle and abs(y - bottom) <= handle:
            return "se"
        return None

    def _start_overlay_interaction(self, overlay: OverlayItem, event: tk.Event) -> None:
        coords = self.canvas.coords(overlay.rect_id)
        if len(coords) != 4:
            return
        left, top, right, bottom = coords
        corner = self._determine_resize_corner((left, top, right, bottom), event.x, event.y)
        if corner is not None:
            self._dragging_mode = "resize"
            self._resize_corner = corner
            if corner == "nw":
                self._resize_anchor = (right, bottom)
            elif corner == "ne":
                self._resize_anchor = (left, bottom)
            elif corner == "sw":
                self._resize_anchor = (right, top)
            else:
                self._resize_anchor = (left, top)
        else:
            self._dragging_mode = "move"
            self._drag_offset = (event.x - left, event.y - top)
            self._resize_anchor = None
            self._resize_corner = None
        self._dragging_overlay = overlay
        self._drag_initial_bbox = (left, top, right, bottom)
        self._drag_original_bbox = overlay.token.bbox
        self._pressed_overlay = None

    def _drag_overlay(self, event: tk.Event) -> None:
        if self._dragging_overlay is None or self._dragging_mode is None:
            return
        overlay = self._dragging_overlay
        if self._drag_initial_bbox is None:
            return
        left, top, right, bottom = self._drag_initial_bbox
        max_width, max_height = self._get_display_bounds()
        min_size = self.MIN_DISPLAY_BOX
        if self._dragging_mode == "move":
            width = right - left
            height = bottom - top
            new_left = event.x - self._drag_offset[0]
            new_top = event.y - self._drag_offset[1]
            if max_width > 0:
                new_left = max(0.0, min(new_left, max_width - width))
            if max_height > 0:
                new_top = max(0.0, min(new_top, max_height - height))
            new_right = new_left + width
            new_bottom = new_top + height
        else:
            if self._resize_anchor is None or self._resize_corner is None:
                return
            anchor_x, anchor_y = self._resize_anchor
            new_right = right
            new_left = left
            new_top = top
            new_bottom = bottom
            if self._resize_corner == "nw":
                new_right = anchor_x
                new_bottom = anchor_y
                new_left = min(event.x, new_right - min_size)
                new_top = min(event.y, new_bottom - min_size)
            elif self._resize_corner == "ne":
                new_left = anchor_x
                new_bottom = anchor_y
                new_right = max(event.x, new_left + min_size)
                new_top = min(event.y, new_bottom - min_size)
            elif self._resize_corner == "sw":
                new_right = anchor_x
                new_top = anchor_y
                new_left = min(event.x, new_right - min_size)
                new_bottom = max(event.y, new_top + min_size)
            else:  # "se"
                new_left = anchor_x
                new_top = anchor_y
                new_right = max(event.x, new_left + min_size)
                new_bottom = max(event.y, new_top + min_size)
            if max_width > 0:
                new_left = max(0.0, new_left)
                new_right = min(max_width, new_right)
                if new_right - new_left < min_size:
                    if self._resize_corner in ("nw", "sw"):
                        new_left = new_right - min_size
                    else:
                        new_right = new_left + min_size
                    new_left = max(0.0, new_left)
                    new_right = min(max_width, new_right)
            if max_height > 0:
                new_top = max(0.0, new_top)
                new_bottom = min(max_height, new_bottom)
                if new_bottom - new_top < min_size:
                    if self._resize_corner in ("nw", "ne"):
                        new_top = new_bottom - min_size
                    else:
                        new_bottom = new_top + min_size
                    new_top = max(0.0, new_top)
                    new_bottom = min(max_height, new_bottom)
        self._update_overlay_display(overlay, (new_left, new_top, new_right, new_bottom))

    def _finalize_overlay_drag(self) -> None:
        if self._dragging_overlay is None:
            return
        overlay = self._dragging_overlay
        coords = self.canvas.coords(overlay.rect_id)
        original_bbox = self._drag_original_bbox
        if len(coords) == 4:
            left, top, right, bottom = coords
            base_bbox = self._display_to_base_bbox((left, top, right, bottom))
            if original_bbox is not None and base_bbox != original_bbox:
                action = UndoAction(kind="transform", transforms=[(overlay.token, original_bbox)])
                self._undo_stack.append(action)
            overlay.token.bbox = base_bbox
            self._position_overlay_handles(overlay, (left, top, right, bottom))
        self._dragging_overlay = None
        self._dragging_mode = None
        self._drag_initial_bbox = None
        self._drag_offset = (0.0, 0.0)
        self._resize_anchor = None
        self._resize_corner = None
        self._drag_original_bbox = None

    def _next_manual_keys(self) -> Tuple[TokenOrder, LineKey]:
        self.manual_token_counter += 1
        index = self.manual_token_counter
        order_key: TokenOrder = (9999, 0, 0, index, index)
        line_key: LineKey = (9999, 0, index)
        return order_key, line_key

    def _create_manual_overlay(self, bbox: Tuple[float, float, float, float]) -> Optional[OverlayItem]:
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

    def _set_box_selected(self, overlay: OverlayItem, selected: bool) -> None:
        overlay.selected = selected
        outline = "#F2994A" if selected else "#2F80ED"
        width = 2 if selected else 1
        try:
            self.canvas.itemconfigure(overlay.rect_id, outline=outline, width=width)
        except tk.TclError:
            return
        if selected:
            self._show_overlay_handles(overlay)
        else:
            self._hide_overlay_handles(overlay)

    def _show_overlay_handles(self, overlay: OverlayItem) -> None:
        if not hasattr(self, "handle_to_overlay"):
            self.handle_to_overlay = {}
        coords = self.canvas.coords(overlay.rect_id)
        if len(coords) != 4:
            return
        left, top, right, bottom = coords
        if not overlay.handles:
            overlay.handles = {}
            for corner in ("nw", "ne", "sw", "se"):
                try:
                    handle_id = self.canvas.create_rectangle(
                        0,
                        0,
                        0,
                        0,
                        outline="white",
                        fill="#F2994A",
                        width=1,
                        tags=("overlay-handle",),
                    )
                except tk.TclError:
                    return
                overlay.handles[corner] = handle_id
                self.handle_to_overlay[handle_id] = (overlay, corner)
        self._position_overlay_handles(overlay, (left, top, right, bottom))
        for handle_id in overlay.handles.values():
            try:
                self.canvas.itemconfigure(handle_id, state="normal")
                self.canvas.tag_raise(handle_id)
            except tk.TclError:
                continue

    def _hide_overlay_handles(self, overlay: OverlayItem) -> None:
        mapping = getattr(self, "handle_to_overlay", None)
        for handle_id in overlay.handles.values():
            try:
                self.canvas.delete(handle_id)
            except tk.TclError:
                pass
            if mapping is not None:
                mapping.pop(handle_id, None)
        overlay.handles.clear()

    def _position_overlay_handles(
        self, overlay: OverlayItem, coords: Optional[Tuple[float, float, float, float]] = None
    ) -> None:
        if not overlay.handles:
            return
        if coords is None:
            coords = self.canvas.coords(overlay.rect_id)
        if len(coords) != 4:
            return
        left, top, right, bottom = coords
        radius = max(4, int(self.RESIZE_HANDLE))
        positions = {
            "nw": (left, top),
            "ne": (right, top),
            "sw": (left, bottom),
            "se": (right, bottom),
        }
        for corner, (x, y) in positions.items():
            handle_id = overlay.handles.get(corner)
            if handle_id is None:
                continue
            try:
                self.canvas.coords(handle_id, x - radius, y - radius, x + radius, y + radius)
                self.canvas.tag_raise(handle_id)
            except tk.TclError:
                continue

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

    def _remove_overlay(self, overlay: OverlayItem) -> None:
        self._hide_overlay_handles(overlay)
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

    def _find_overlay_at(self, x: float, y: float) -> Optional[OverlayItem]:
        for overlay in reversed(self.overlay_items):
            coords = self.canvas.coords(overlay.rect_id)
            if len(coords) != 4:
                continue
            left, top, right, bottom = coords
            if left <= x <= right and top <= y <= bottom:
                return overlay
        return None

    def _overlays_in_rect(self, bbox: Tuple[float, float, float, float]) -> List[OverlayItem]:
        left, top, right, bottom = bbox
        selected: list[OverlayItem] = []
        for overlay in self.overlay_items:
            coords = self.canvas.coords(overlay.rect_id)
            if len(coords) != 4:
                continue
            o_left, o_top, o_right, o_bottom = coords
            if o_left >= right or o_right <= left or o_top >= bottom or o_bottom <= top:
                continue
            selected.append(overlay)
        return selected

    def _apply_single_selection(self, overlay: OverlayItem, additive: bool) -> None:
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

    def _apply_marquee_selection(self, overlays: Sequence[OverlayItem], additive: bool) -> None:
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
        self._pressed_overlay = None
        self._marquee_rect = None
        self._marquee_dragging = False
        self._marquee_additive = False
        self._dragging_overlay = None
        self._dragging_mode = None
        self._drag_initial_bbox = None
        self._resize_anchor = None
        self._resize_corner = None
        self._drag_original_bbox = None
        mode = self._get_mode()
        if mode == "zoom":
            step = self.ZOOM_STEP
            if self._event_has_ctrl(event) or self._event_has_shift(event):
                step = 1 / self.ZOOM_STEP
            focus = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            self._apply_zoom(self.zoom_factor * step, focus=focus)
            self._drag_start = None
            return
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
            return
        handle_target: Optional[Tuple[OverlayItem, str]] = None
        find_withtag = getattr(self.canvas, "find_withtag", None)
        if callable(find_withtag):
            current = find_withtag("current")
            if current:
                handle_target = self.handle_to_overlay.get(current[0])
        if handle_target is not None:
            overlay, corner = handle_target
            if overlay is None:
                return
            if overlay.rect_id not in self.selected_rects:
                self._apply_single_selection(overlay, additive=False)
            self._start_overlay_interaction(overlay, event)
            return

        additive = self._event_has_ctrl(event) or self._event_has_shift(event)
        canvasx = getattr(self.canvas, "canvasx", None)
        canvasy = getattr(self.canvas, "canvasy", None)
        canvas_x = canvasx(event.x) if callable(canvasx) else event.x
        canvas_y = canvasy(event.y) if callable(canvasy) else event.y
        overlay = self._find_overlay_at(canvas_x, canvas_y)
        if overlay is not None:
            self._pressed_overlay = overlay
            if not additive and overlay.rect_id in self.selected_rects:
                self._start_overlay_interaction(overlay, event)
            return

        self._pressed_overlay = None
        self._marquee_dragging = True
        self._marquee_additive = additive
        if not additive:
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
            if self._dragging_overlay is not None:
                self._drag_overlay(event)
                return
            if not self._marquee_dragging:
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
            if self._dragging_overlay is not None:
                self._finalize_overlay_drag()
            elif self._marquee_rect is not None:
                coords = self.canvas.coords(self._marquee_rect)
                self.canvas.delete(self._marquee_rect)
                if len(coords) == 4:
                    left, top, right, bottom = coords
                    bbox = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))
                    overlays = self._overlays_in_rect(bbox)
                    additive = self._marquee_additive
                    self._apply_marquee_selection(overlays, additive=additive)
            else:
                overlay = self._pressed_overlay
                if overlay is not None:
                    additive = self._event_has_ctrl(event) or self._event_has_shift(event)
                    if additive or overlay.rect_id not in self.selected_rects:
                        self._apply_single_selection(overlay, additive=additive)
                elif not self._marquee_additive:
                    self._clear_selection()
        self._pressed_overlay = None
        self._drag_start = None
        self._marquee_rect = None
        self._marquee_dragging = False
        self._marquee_additive = False

    def _on_canvas_mousewheel(self, event: tk.Event) -> str:
        direction = 0
        delta = getattr(event, "delta", 0)
        if delta > 0:
            direction = 1
        elif delta < 0:
            direction = -1
        else:
            num = getattr(event, "num", 0)
            if num == 4:
                direction = 1
            elif num == 5:
                direction = -1
        if direction == 0:
            return ""
        step = self.ZOOM_STEP if direction > 0 else 1 / self.ZOOM_STEP
        focus = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self._apply_zoom(self.zoom_factor * step, focus=focus)
        return "break"

    def _apply_zoom(
        self,
        target: float,
        *,
        focus: Optional[Tuple[float, float]] = None,
    ) -> None:
        if self._base_display_image is None:
            return
        target = max(self.MIN_ZOOM, min(self.MAX_ZOOM, target))
        if abs(target - self.zoom_factor) < 1e-3:
            return
        old_factor = self.zoom_factor
        base_width, base_height = self._base_display_image.size
        new_width = max(1, int(base_width * target))
        new_height = max(1, int(base_height * target))
        resized = self._base_display_image.resize((new_width, new_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        self.current_photo = photo
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(0, 0, image=photo, anchor="nw")
        else:
            try:
                self.canvas.itemconfigure(self.canvas_image_id, image=photo)
            except tk.TclError:
                self.canvas_image_id = self.canvas.create_image(0, 0, image=photo, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        self.zoom_factor = target
        base_scale_x, base_scale_y = self._base_scale
        self.display_scale = (base_scale_x * target, base_scale_y * target)
        self._update_overlay_positions()
        if focus is not None:
            self._adjust_canvas_view(focus, old_factor, target, new_width, new_height)

    def _adjust_canvas_view(
        self,
        focus: Tuple[float, float],
        old_factor: float,
        new_factor: float,
        new_width: int,
        new_height: int,
    ) -> None:
        if old_factor <= 0:
            scale_ratio = new_factor
        else:
            scale_ratio = new_factor / old_factor
        new_focus_x = focus[0] * scale_ratio
        new_focus_y = focus[1] * scale_ratio
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        if new_width > canvas_width:
            left = max(0.0, min(new_focus_x - canvas_width / 2, new_width - canvas_width))
            self.canvas.xview_moveto(left / new_width)
        else:
            self.canvas.xview_moveto(0)
        if new_height > canvas_height:
            top = max(0.0, min(new_focus_y - canvas_height / 2, new_height - canvas_height))
            self.canvas.yview_moveto(top / new_height)
        else:
            self.canvas.yview_moveto(0)

    def _update_overlay_positions(self) -> None:
        scale_x, scale_y = self.display_scale
        for overlay in self.overlay_items:
            left, top, right, bottom = overlay.token.bbox
            disp_left = left * scale_x
            disp_top = top * scale_y
            disp_right = right * scale_x
            disp_bottom = bottom * scale_y
            try:
                self.canvas.coords(overlay.rect_id, disp_left, disp_top, disp_right, disp_bottom)
            except tk.TclError:
                continue
            width = max(4, int(max(1, abs(disp_right - disp_left)) / 8))
            try:
                overlay.entry.configure(width=width)
            except tk.TclError:
                pass
            desired_top = disp_top - 24
            if desired_top < 0:
                desired_top = disp_top
            try:
                self.canvas.coords(overlay.window_id, disp_left, desired_top)
            except tk.TclError:
                continue
            self._position_overlay_handles(overlay, (disp_left, disp_top, disp_right, disp_bottom))

    def _add_overlay_item(
        self,
        token: OcrToken,
        text: str,
        index: Optional[int] = None,
        scale: Optional[Tuple[float, float]] = None,
    ) -> None:
        left, top, right, bottom = token.bbox
        scale_x, scale_y = scale if scale is not None else self.display_scale
        disp_left = left * scale_x
        disp_top = top * scale_y
        disp_right = right * scale_x
        disp_bottom = bottom * scale_y

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

        self.rect_to_overlay[rect] = item
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
            removed.append(RemovedOverlay(token=item.token, text=item.entry.get(), index=index, scale=self.display_scale))
            if item.fade_job is not None:
                try:
                    self.master.after_cancel(item.fade_job)
                except tk.TclError:
                    pass
                item.fade_job = None
            self._hide_overlay_handles(item)
            try:
                self.canvas.delete(item.rect_id)
                self.canvas.delete(item.window_id)
            except tk.TclError:
                pass
            try:
                item.entry.destroy()
            except tk.TclError:
                pass
            self.rect_to_overlay.pop(item.rect_id, None)
            del self.overlay_items[index]
            del self.current_tokens[index]

        if not removed:
            return False

        # Maintain original order for undo restoration.
        removed.sort(key=lambda overlay: overlay.index)
        self._undo_stack.append(UndoAction(kind="delete", overlays=removed))
        self._update_combined_transcription()
        return True

    def _on_undo(self, event: Optional[tk.Event]) -> str:
        if not self._undo_stack:
            return ""
        action = self._undo_stack.pop()
        if action.kind == "delete":
            for overlay in action.overlays:
                self._add_overlay_item(overlay.token, overlay.text, index=overlay.index)
        elif action.kind == "transform":
            for token, bbox in action.transforms:
                token.bbox = bbox
                overlay = self._find_overlay(token)
                if overlay is None:
                    continue
                disp_left, disp_top, disp_right, disp_bottom = self._base_to_display_bbox(bbox)
                self._update_overlay_display(
                    overlay,
                    (disp_left, disp_top, disp_right, disp_bottom),
                )
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

    def _add_overlay_item(
        self,
        token: OcrToken,
        text: str,
        index: Optional[int] = None,
        scale: Optional[Tuple[float, float]] = None,
    ) -> None:
        left, top, right, bottom = token.bbox
        scale_x, scale_y = scale if scale is not None else self.display_scale
        disp_left = left * scale_x
        disp_top = top * scale_y
        disp_right = right * scale_x
        disp_bottom = bottom * scale_y

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

        self.rect_to_overlay[rect] = item
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
            removed.append(RemovedOverlay(token=item.token, text=item.entry.get(), index=index, scale=self.display_scale))
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

    def _add_overlay_item(
        self,
        token: OcrToken,
        text: str,
        index: Optional[int] = None,
        scale: Optional[Tuple[float, float]] = None,
    ) -> None:
        left, top, right, bottom = token.bbox
        scale_x, scale_y = scale if scale is not None else self.display_scale
        disp_left = left * scale_x
        disp_top = top * scale_y
        disp_right = right * scale_x
        disp_bottom = bottom * scale_y

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

        self.rect_to_overlay[rect] = item
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
            removed.append(RemovedOverlay(token=item.token, text=item.entry.get(), index=index, scale=self.display_scale))
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

    def _add_overlay_item(
        self,
        token: OcrToken,
        text: str,
        index: Optional[int] = None,
        scale: Optional[Tuple[float, float]] = None,
    ) -> None:
        left, top, right, bottom = token.bbox
        scale_x, scale_y = scale if scale is not None else self.display_scale
        disp_left = left * scale_x
        disp_top = top * scale_y
        disp_right = right * scale_x
        disp_bottom = bottom * scale_y

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

        self.rect_to_overlay[rect] = item
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
            removed.append(RemovedOverlay(token=item.token, text=item.entry.get(), index=index, scale=self.display_scale))
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

    def _add_overlay_item(
        self,
        token: OcrToken,
        text: str,
        index: Optional[int] = None,
        scale: Optional[Tuple[float, float]] = None,
    ) -> None:
        left, top, right, bottom = token.bbox
        scale_x, scale_y = scale if scale is not None else self.display_scale
        disp_left = left * scale_x
        disp_top = top * scale_y
        disp_right = right * scale_x
        disp_bottom = bottom * scale_y

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

        self.rect_to_overlay[rect] = item
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
            removed.append(RemovedOverlay(token=item.token, text=item.entry.get(), index=index, scale=self.display_scale))
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
