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
* The confirmed transcription is also written to
  ``transcripts/raw/<image>.txt`` (configurable via ``--transcripts-dir``) so
  that subsequent packaging steps include your corrections.
* Images are automatically rotated based on their embedded EXIF orientation so
  the preview and saved snippet share the correct layout.
* Use **Skip** to omit an image or **Unsure** to log it without saving a copy.
  The optional `--output-log` CSV records every action with `page`,
  `transcription`, and `timestamp` columns for downstream tooling.
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

#### Locating Tesseract's tessdata files

Tesseract needs access to its language packs (the `tessdata` folder) before the
training pipeline can start. The CLI now attempts to discover this directory in
several ways:

1. Respect an explicit `--tessdata-dir` argument when you provide one.
2. Fall back to the `TESSDATA_PREFIX` environment variable if it is set.
3. Ask the local Tesseract binary via `tesseract --print-tessdata-dir`.
4. Check the default installation paths on Windows and Linux.

If all of these checks fail you'll see an error similar to "Unable to locate
tessdata directory". Fix it by either setting the environment variable or
passing the path explicitly:

```bash
# macOS / Linux
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Windows PowerShell (default install location)
$env:TESSDATA_PREFIX="C:/Program Files/Tesseract-OCR/tessdata"

# Or pass it directly to the CLI
python main.py train --train-dir train --output-model handwriting \
    --tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"
```

> **Tip:** Running `tesseract --print-tessdata-dir` in your shell prints the
> directory that the binary uses internally. Point `TESSDATA_PREFIX` to that
> location if the training command cannot find it automatically.

### ChatGPT-powered transcription

Set the `OPENAI_API_KEY` environment variable to enable ChatGPT's multimodal
API for handwriting transcription. When active, the training pipeline sends
each snippet to ChatGPT, writes the recognised text into the required
`.gt.txt` files, and then proceeds with the normal Tesseract fine-tuning
workflow. The default configuration uses the `gpt-4o-mini` model and caches can
optionally be persisted via `--gpt-cache-dir` to avoid duplicate API calls.

You can configure the key either by exporting it in your shell **or** by
creating a `.env` file in the project root (a `.env.example` template is
included and the CLI now loads this file automatically):

```bash
# macOS / Linux shell
export OPENAI_API_KEY="sk-your-key"

# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key"

# Optional .env file if you prefer to keep credentials out of your shell profile
echo "OPENAI_API_KEY=sk-your-key" > .env
```

Existing environment variables take precedence over the `.env` file so you can
override the key per-shell or per-command when needed.

To customise the behaviour use the new CLI flags, for example:

```bash
python main.py train --train-dir train --output-model handwriting \
    --gpt-model gpt-4o --gpt-cache-dir .cache/gpt
```

Pass `--gpt-max-images N` to cap how many samples are transcribed by ChatGPT
before the pipeline falls back to file-name derived labels. This helps limit API
usage during experimentation.

Pass `--no-gpt-ocr` to fall back to the legacy behaviour of deriving labels from
file names when required.

> **Note:** The annotation interface relies on Tkinter, which ships with most
> standard Python installers. On some Linux distributions you may need to
> install an additional package such as `python3-tk` to enable the GUI.

## 🧠 TrOCR Mode (Handwriting OCR)

Standup-OCR now ships with [Microsoft's TrOCR](https://huggingface.co/microsoft)
handwriting model via Hugging Face Transformers. Install the dependencies with:

```bash
pip install torch torchvision transformers pillow
```

Run the automated handwriting pipeline with:

```bash
python main.py annotate --source path/to/folder --output-log train/annotation_log.csv \
    --force --verbose
```

The agentic helper processes each scan with TrOCR, writes transcripts to
`transcripts/raw/`, and records page-level results in `train/annotation_log.csv`
with the columns `page`, `transcription`, and `timestamp`. A summary of the run
is printed as `[TrOCR] Processed …` and `ready_for_review.zip` is refreshed with
the newly generated text files.

Use the `--model` flag on `python -m src.ocr.main` to experiment with other
TrOCR variants (for example `microsoft/trocr-large-handwritten`).

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
