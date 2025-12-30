"""Quick Tkinter viewer for inspecting full-page segmentation overlays from Kraken.

This viewer loads PAGE-XML boxes (or .boxes.json metadata) and draws all detected
line boundaries and baselines on the original page image, allowing you to visually
verify segmentation accuracy without manually checking individual crop files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageTk

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class OverlayViewer:
    """Viewer that displays segmentation overlays (boxes, baselines) on original page images."""

    def __init__(self, metadata_paths: List[Path]) -> None:
        """Initialize with paths to .boxes.json files (or directories containing them).
        
        Args:
            metadata_paths: List of .boxes.json files or directories to scan.
        """
        if not metadata_paths:
            raise ValueError("No metadata files supplied to the overlay viewer.")
        
        # Group metadata by source_image so we can display all lines on one page
        self.pages: dict[str, list[dict]] = {}
        self.page_paths: List[str] = []
        
        for meta_path in metadata_paths:
            if meta_path.is_file() and meta_path.suffix == ".json":
                self._load_metadata(meta_path)
            elif meta_path.is_dir():
                for json_file in sorted(meta_path.glob("*.boxes.json")):
                    self._load_metadata(json_file)
        
        if not self.pages:
            raise ValueError("No .boxes.json metadata files found.")
        
        self.page_paths = sorted(self.pages.keys())
        self.index = 0

        self.root = tk.Tk()
        self.root.title("Standup-OCR Full-Page Overlay Viewer")
        self.canvas = tk.Canvas(self.root, background="#1e1e1e")
        self.canvas.pack(fill="both", expand=True)

        self.status_var = tk.StringVar()
        status = tk.Label(self.root, textvariable=self.status_var, anchor="w")
        status.pack(fill="x")

        self.root.bind("<Left>", lambda _e: self._step(-1))
        self.root.bind("<Right>", lambda _e: self._step(1))
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

        self._photo: ImageTk.PhotoImage | None = None
        self._render()

    def _load_metadata(self, meta_path: Path) -> None:
        """Load a .boxes.json file and group by source_image."""
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: Failed to load {meta_path}: {e}", file=sys.stderr)
            return
        
        source_image = data.get("source_image")
        if not source_image:
            print(f"Warning: {meta_path} has no 'source_image' field", file=sys.stderr)
            return
        
        if source_image not in self.pages:
            self.pages[source_image] = []
        self.pages[source_image].append(data)

    def _step(self, delta: int) -> None:
        """Navigate to previous/next page."""
        self.index = (self.index + delta) % len(self.page_paths)
        self._render()

    def _render(self) -> None:
        """Render the current page with all line overlays."""
        source_path = self.page_paths[self.index]
        metadata_list = self.pages[source_path]
        
        # Try to open the source image
        source_image_path = Path(source_path)
        if not source_image_path.exists():
            # Try relative to the workspace root
            source_image_path = ROOT / source_path
        
        if not source_image_path.exists():
            self.status_var.set(
                f"SOURCE NOT FOUND: {source_path} "
                f"({self.index + 1}/{len(self.page_paths)})"
            )
            self.canvas.delete("all")
            return
        
        # Load and prepare the base image
        try:
            with Image.open(source_image_path) as base_img:
                # Apply EXIF orientation to rotate image upright
                display = ImageOps.exif_transpose(base_img)
                display = display.convert("RGB")
        except Exception as e:
            self.status_var.set(f"Failed to load image: {e}")
            self.canvas.delete("all")
            return
        
        # Downscale large images for display (max 1200 pixels on longest side)
        max_display_dim = 1200
        orig_width, orig_height = display.size
        scale_factor = 1.0
        
        if max(orig_width, orig_height) > max_display_dim:
            if orig_width > orig_height:
                scale_factor = max_display_dim / orig_width
            else:
                scale_factor = max_display_dim / orig_height
            
            new_width = int(orig_width * scale_factor)
            new_height = int(orig_height * scale_factor)
            display = display.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Draw all line overlays
        draw = ImageDraw.Draw(display, "RGBA")
        
        for metadata in metadata_list:
            # Draw bounding box using denormalized coordinates if available, else normalized
            bbox_data = metadata.get("bbox_original") or metadata.get("bbox")
            if bbox_data and isinstance(bbox_data, dict):
                left = int(bbox_data.get("left", 0) * scale_factor)
                top = int(bbox_data.get("top", 0) * scale_factor)
                right = int(bbox_data.get("right", 0) * scale_factor)
                bottom = int(bbox_data.get("bottom", 0) * scale_factor)
                # Red rectangle with transparency
                draw.rectangle(
                    [(left, top), (right, bottom)],
                    outline=(239, 68, 68, 255),
                    width=2,
                )
            
            # Draw baseline if available
            baseline_data = metadata.get("baseline_original") or metadata.get("baseline")
            if baseline_data and isinstance(baseline_data, list) and len(baseline_data) >= 2:
                try:
                    # baseline_data should be a list of [x, y] pairs
                    points = []
                    for point in baseline_data:
                        if isinstance(point, (list, tuple)) and len(point) == 2:
                            scaled_x = int(point[0] * scale_factor)
                            scaled_y = int(point[1] * scale_factor)
                            points.append((scaled_x, scaled_y))
                    
                    if len(points) >= 2:
                        # Cyan polyline with transparency
                        for i in range(len(points) - 1):
                            draw.line(
                                [points[i], points[i + 1]],
                                fill=(56, 189, 248, 255),
                                width=2,
                            )
                except Exception as e:
                    print(f"Warning: Failed to draw baseline: {e}", file=sys.stderr)
        
        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(display)
        self._photo = photo
        self.canvas.delete("all")
        self.canvas.config(width=photo.width(), height=photo.height())
        self.canvas.create_image(0, 0, image=photo, anchor="nw")
        
        # Update status
        line_count = len(metadata_list)
        scale_pct = int(scale_factor * 100)
        self.status_var.set(
            f"{source_image_path.name} ({line_count} lines, {scale_pct}% scale) "
            f"({self.index + 1}/{len(self.page_paths)})"
        )

    def run(self) -> None:
        """Start the viewer's event loop."""
        self.root.mainloop()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Directories or .boxes.json files to preview. Default: train/kraken_auto_lines_small",
    )
    parser.add_argument(
        "--glob",
        default="*.boxes.json",
        help="File pattern for metadata (default: *.boxes.json).",
    )
    return parser


def collect_metadata_files(paths: List[Path], pattern: str) -> List[Path]:
    """Collect all metadata files matching the pattern from the given paths."""
    metadata_files: List[Path] = []
    for path in paths:
        if path.is_dir():
            metadata_files.extend(sorted(path.glob(pattern)))
        elif path.is_file() and path.suffix == ".json":
            metadata_files.append(path)
    return metadata_files


def main(argv: List[str] | None = None) -> None:
    """Main entry point for the overlay viewer."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    
    # Default to kraken_auto_lines_small if no paths provided
    inputs = args.paths or [ROOT / "train" / "kraken_auto_lines_small"]
    metadata_files = collect_metadata_files(inputs, args.glob)
    
    if not metadata_files:
        raise SystemExit(
            f"No .boxes.json files matched the provided arguments.\n"
            f"Searched in: {inputs}"
        )
    
    viewer = OverlayViewer(metadata_files)
    viewer.run()


if __name__ == "__main__":
    main()
