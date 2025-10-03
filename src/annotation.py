"""GUI for manually annotating handwriting samples."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageOps, ImageTk


def _prepare_image(image: Image.Image) -> Image.Image:
    """Return an image with EXIF orientation applied."""

    return ImageOps.exif_transpose(image)


@dataclass
class AnnotationItem:
    """Represent a single image queued for annotation."""

    path: Path


class AnnotationApp:
    """Tkinter-based interface for stepping through a set of images."""

    MAX_SIZE = (900, 700)

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

        self.filename_var = tk.StringVar()
        self.entry_var = tk.StringVar()
        self.status_var = tk.StringVar()

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

        self.image_label = tk.Label(container, bd=1, relief="sunken")
        self.image_label.pack(fill="both", expand=True, pady=12)

        entry_frame = tk.Frame(container)
        entry_frame.pack(fill="x", pady=(0, 8))

        tk.Label(entry_frame, text="Transcription:").pack(side="left")
        entry = tk.Entry(entry_frame, textvariable=self.entry_var, width=50)
        entry.pack(side="left", fill="x", expand=True, padx=(8, 0))
        entry.bind("<Return>", self._on_confirm)
        self.entry_widget = entry

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
        self.master.protocol("WM_DELETE_WINDOW", self._on_exit)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_confirm(self, event: tk.Event | None) -> None:
        self.confirm()

    def _on_exit(self, event: tk.Event | None = None) -> None:
        if messagebox.askokcancel("Quit", "Abort annotation and close the window?"):
            self.master.destroy()

    def confirm(self) -> None:
        label = self.entry_var.get().strip()
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
        self._append_log(item.path, self.entry_var.get().strip(), "unsure", None)
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
        self.entry_var.set(self._suggest_label(item.path))
        self._display_image(item.path)
        self.entry_widget.selection_range(0, tk.END)
        self.entry_widget.focus_set()

    def _display_image(self, path: Path) -> None:
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

        self.current_photo = photo
        self.image_label.configure(image=photo)
        self.image_label.image = photo

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
            image = _prepare_image(image)
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            image.save(candidate)
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
