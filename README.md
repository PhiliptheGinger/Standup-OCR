# Standup-OCR

Standup-OCR is a small toolkit that helps capture handwritten stand-up notes,
correct OCR mistakes, and fine-tune a custom Tesseract model on the confirmed
samples.

## Usage

```bash
# Core OCR run
python -m src.ocr.main --agentic

# Re-run missing files
python -m src.ocr.main --rerun-failed

# Check missing scans
python -m src.ocr.utils.pending_scans

# Verify logs
python -m src.verify.annotation_log --check

# Upload for GPT transcription
python -m src.ocr.upload_to_gpt

# Continuous background mode
python -m src.agent.watchdog

# Optional Drive sync
python -m src.sync.drive_upload
```

## Command-line interface

Install the dependencies listed in `requirements.txt`, then invoke the CLI via

```bash
python main.py --help
```

### Reviewing OCR output

Use the `review` subcommand to step through low-confidence tokens produced by
`ocr_detailed`. The tool crops each region from the preprocessed image and asks
for a corrected transcription. Confirmed snippets are written to the training
directory (`train/` by default) using the `<prefix>_<label>.png` naming
convention so they can be consumed by the existing training pipeline.

```bash
python main.py review --source path/to/image_or_folder --threshold 75 \
    --train-dir train --auto-train 10
```

Key behaviour:

* `--source` accepts either a single image or a folder containing images.
* `--threshold` controls which OCR tokens require review (defaults to 70).
* `--train-dir` points to the folder where confirmed snippets will be saved.
* `--auto-train N` triggers `train_model` once every `N` newly confirmed
  samples, reusing the options passed to the command.
* Use `--no-preview` on headless systems to disable image previews. Otherwise
  the tool attempts to display each snippet using `PIL.Image.show()` and falls
  back to logging the path of a temporary PNG file when required.

During review you can:

* Press **Enter** to accept the recognised text (when available).
* Type a corrected transcription to save it under that label.
* Enter `s` to skip a snippet or `q` to end the session early.

Each confirmed snippet is recorded in `train/review_log.jsonl`, preventing the
same bounding box from being queued again in future sessions.

### Annotating full-page images

Use the `annotate` subcommand when you already have a folder of scans that need
verified transcriptions. The tool opens a small Tkinter window, displays each
image, and lets you enter the ground-truth text before saving a copy into your
training directory using the `<prefix>_<label>.png` naming convention.

```bash
python main.py annotate --source path/to/folder --train-dir train \
    --output-log train/annotation_log.csv
```

Key behaviour:

* `--source` accepts a single image or a directory of images with supported
  extensions.
* Confirming an entry copies the image into `--train-dir` with the confirmed
  label embedded in the file name, ready for `train_model`.
* Images are automatically rotated based on their embedded EXIF orientation so
  the preview and saved snippet share the correct layout.
* Use **Skip** to omit an image or **Unsure** to log it without saving a copy.
  The optional `--output-log` CSV records every action.
* The preview honours EXIF orientation so sideways scans load upright, and the
  OCR suggestion is rendered as editable overlays directly above each detected
  word for quick correction before saving.

### Training a custom model

Once you have collected a set of labelled snippets run:

```bash
python main.py train --train-dir train --output-model handwriting
```

Use the `--model` flag on the `test`, `batch`, or `review` subcommands to
evaluate the updated model.

> **Note:** The annotation interface relies on Tkinter, which ships with most
> standard Python installers. On some Linux distributions you may need to
> install an additional package such as `python3-tk` to enable the GUI.

## Kraken installation (Windows quick start)

1. Install [Python 3.10+](https://www.python.org/downloads/windows/) and ensure "Add python.exe to PATH" is ticked during setup.
2. Open **PowerShell** and install [pipx](https://pypa.github.io/pipx/) if it is not already available:

   ```powershell
   python -m pip install --user pipx
   python -m pipx ensurepath
   ```

3. Close and reopen PowerShell, then install Kraken together with its CLI tools:

   ```powershell
   pipx install "kraken[serve]"
   ```

4. Confirm the installation with `kraken --version` and `ketos --help`. Both commands should be available in any new shell session.
