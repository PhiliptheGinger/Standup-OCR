"""Command-line interface for the handwriting OCR toolkit."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.ocr import ocr_image
from src.training import SUPPORTED_EXTENSIONS, train_model


DEFAULT_TRAIN_DIR = Path("train")
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_RESULTS_FILE = Path("results.csv")


def setup_logging(verbose: bool = False) -> None:
    """Configure the root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def iter_images(folder: Path) -> Iterable[Path]:
    """Yield image files from ``folder`` that match supported extensions."""
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def handle_train(args: argparse.Namespace) -> None:
    model_path = train_model(
        args.train_dir,
        args.output_model,
        model_dir=args.model_dir,
        tessdata_dir=args.tessdata_dir,
        base_lang=args.base_lang,
        max_iterations=args.max_iterations,
    )
    logging.info("Model saved to %s", model_path)


def handle_test(args: argparse.Namespace) -> None:
    text = ocr_image(
        args.image,
        model_path=args.model,
        tessdata_dir=args.tessdata_dir,
        psm=args.psm,
    )
    print(text)


def handle_batch(args: argparse.Namespace) -> None:
    folder = Path(args.folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows: List[dict[str, str]] = []
    for image_path in iter_images(folder):
        try:
            text = ocr_image(
                image_path,
                model_path=args.model,
                tessdata_dir=args.tessdata_dir,
                psm=args.psm,
            )
        except Exception as exc:  # pragma: no cover - runtime logging only
            logging.exception("Failed to OCR %s", image_path)
            text = f"ERROR: {exc}"
        rows.append({"image": image_path.name, "text": text})

    df = pd.DataFrame(rows)
    output = Path(args.output)
    df.to_csv(output, index=False)
    logging.info("Batch OCR complete. Results saved to %s", output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Fine-tune a Tesseract model using samples placed in the train/ folder.",
    )
    train_parser.add_argument(
        "--train-dir",
        type=Path,
        default=DEFAULT_TRAIN_DIR,
        help="Directory with training images (default: train/).",
    )
    train_parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Where to store trained models (default: models/).",
    )
    train_parser.add_argument(
        "--output-model",
        default="handwriting",
        help="Base name of the output model (default: handwriting).",
    )
    train_parser.add_argument(
        "--tessdata-dir",
        type=Path,
        help="Path to tessdata directory containing base traineddata files.",
    )
    train_parser.add_argument(
        "--base-lang",
        default="eng",
        help="Base language code to fine-tune (default: eng).",
    )
    train_parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Training iterations to run (default: 1000).",
    )
    train_parser.set_defaults(func=handle_train)

    test_parser = subparsers.add_parser(
        "test",
        help="Run OCR on a single image and print the recognised text.",
    )
    test_parser.add_argument("image", type=Path, help="Path to the image to OCR.")
    test_parser.add_argument(
        "--model",
        type=Path,
        help="Optional custom .traineddata model to use (defaults to eng).",
    )
    test_parser.add_argument(
        "--tessdata-dir",
        type=Path,
        help="Directory containing tessdata files (defaults to model's folder).",
    )
    test_parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (default: 6).",
    )
    test_parser.set_defaults(func=handle_test)

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run OCR on all images inside a folder and export a CSV report.",
    )
    batch_parser.add_argument("folder", type=Path, help="Folder of images to OCR.")
    batch_parser.add_argument(
        "--model",
        type=Path,
        help="Optional custom .traineddata model to use for batch OCR.",
    )
    batch_parser.add_argument(
        "--tessdata-dir",
        type=Path,
        help="Directory containing tessdata files (defaults to model's folder).",
    )
    batch_parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (default: 6).",
    )
    batch_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="Where to save the CSV summary (default: results.csv).",
    )
    batch_parser.set_defaults(func=handle_batch)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
