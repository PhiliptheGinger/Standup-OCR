def read_gt_txt(images_dir, file_number):
    for ext in ("gt.txt", "gttext", "gt"):  # support common variants
        for fmt in (f"{file_number:03}.{ext}", f"{file_number}.{ext}"):
            path = os.path.join(images_dir, fmt)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    lines = [line.rstrip('\n') for line in f]
                return lines, path
    return None, None
"""
segment_lines_by_page.py

Utility to reorganize standup_option_c_indexed.json by true visual line breaks using image segmentation and OCR.
- Uses PNGs in data/train/images (not PDFs).
- Integrates ChatGPT for post-OCR cleanup.

Usage:
    python scripts/segment_lines_by_page.py --input transcripts/raw/standup_option_c_indexed.json --images-dir data/train/images --output transcripts/raw/standup_option_c_linebyline.json [--openai-api-key <key>]

"""
import os
import json
import argparse
from glob import glob
from typing import List, Dict, Any


import cv2
import numpy as np
import pytesseract
from pathlib import Path

try:
    import openai
except ImportError:
    openai = None


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj: Any, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def find_image_for_page(images_dir: str, file_number: int) -> str:
    # Try zero-padded and non-padded
    candidates = []
    for ext in ("jpg", "jpeg", "png", "tif", "tiff"):
        for fmt in (f"{file_number:03}.{ext}", f"{file_number}.{ext}"):
            path = os.path.join(images_dir, fmt)
            if os.path.exists(path):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No image found for file_number {file_number} in {images_dir}")
    return sorted(candidates)[0]

def preprocess_image(img):
    # Grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    # Median blur
    img = cv2.medianBlur(img, 3)
    return img

def segment_lines_geometric(image_path: str, debug_dir=None, file_number=None):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        print(f"Warning: Could not load image or image is empty: {image_path}")
        return []
    img = preprocess_image(img)
    # Adaptive threshold
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Horizontal dilation
    W = max(10, int(img.shape[1] * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W, 1))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    # Find contours (lines)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bands = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 8 or w < 30:
            continue  # filter noise
        bands.append({'x_left': int(x), 'x_right': int(x+w), 'y_top': int(y), 'y_bottom': int(y+h)})
    # Sort by y, then x
    bands = sorted(bands, key=lambda b: (b['y_top'] + b['y_bottom'])/2)
    # Debug overlay
    if debug_dir and file_number is not None:
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i, b in enumerate(bands):
            cv2.rectangle(overlay, (b['x_left'], b['y_top']), (b['x_right'], b['y_bottom']), (0,255,0), 2)
            cv2.putText(overlay, str(i+1), (b['x_left'], b['y_top']-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"{file_number:03}_annotated.jpg"), overlay)
    return bands

def ocr_band(img, band):
    pad = 5
    y0 = max(0, band['y_top']-pad)
    y1 = min(img.shape[0], band['y_bottom']+pad)
    x0 = max(0, band['x_left']-pad)
    x1 = min(img.shape[1], band['x_right']+pad)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        print(f"[OCR DEBUG] Skipping empty/invalid crop: shape={crop.shape} band={band}")
        return "", 0.0
    config = '--psm 7'
    try:
        data = pytesseract.image_to_data(crop, config=config, lang='eng', output_type=pytesseract.Output.DICT)
        text = " ".join([w for w in data['text'] if w.strip()])
        confs = [float(c) for c in data['conf'] if c != '-1']
        conf = float(np.mean(confs)) if confs else 0.0
        print(f"[OCR DEBUG] Crop shape={crop.shape} conf={conf:.1f} text='{text.strip()[:40]}'")
        return text.strip(), conf
    except Exception as e:
        print(f"[OCR ERROR] {e} for band {band} crop shape={crop.shape}")
        return "", 0.0

def classify_line(band, left_margins, text):
    margin = band['x_left']
    dominant = int(np.median(left_margins)) if left_margins else margin
    if text.lstrip().startswith(('-', '→', '↳')) and abs(margin - dominant) < 10:
        return 'sub'
    if margin > dominant + 15:
        return 'sub'
    return 'main'


def chatgpt_vision_classify_and_transcribe(crop, openai_api_key=None):
    """
    Use GPT-4 Vision to classify if the crop is a real handwritten text line and optionally transcribe it.
    Returns (is_text_line: bool, transcription: str)
    """
    if not openai or not openai_api_key:
        return False, ""
    import base64
    import io
    from PIL import Image
    # Convert crop to JPEG bytes
    pil_img = Image.fromarray(crop)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    vision_prompt = (
        "You are a vision model for OCR post-processing. Given the following image crop, answer strictly 'yes' or 'no': Does this crop contain a real, complete handwritten text line (not just a mark, underline, symbol, or noise)? "
        "If yes, also provide your best transcription of the text line.\n\n"
        "Respond in JSON: {\"is_text_line\": true/false, \"transcription\": ""}"
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
                ]}
            ],
            max_tokens=128,
            temperature=0.2,
        )
        import json as _json
        content = resp['choices'][0]['message']['content']
        # Try to parse JSON from response
        try:
            result = _json.loads(content)
            is_text_line = bool(result.get("is_text_line", False))
            transcription = result.get("transcription", "").strip()
        except Exception:
            # Fallback: try to parse manually
            is_text_line = 'true' in content.lower()
            transcription = ""
        return is_text_line, transcription
    except Exception as e:
        print(f"ChatGPT Vision error: {e}")
        return False, ""

def process_page(image_path, file_number=None, debug_dir=None, openai_api_key=None, gt_lines=None):
    # If GT exists, GT IS THE TRUTH for line breaks. Do not OCR for text.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    bands = segment_lines_geometric(image_path, debug_dir=debug_dir, file_number=file_number)

    # If no GT, fall back to OCR (unchanged)
    if not gt_lines:
        left_margins = [b['x_left'] for b in bands]
        out = []
        for band in bands:
            text, conf = ocr_band(img, band)
            if not text.strip():
                continue
            kind = classify_line(band, left_margins, text)
            out.append({
                "arrow": kind,
                "text": text,
                "bbox": [band['x_left'], band['y_top'], band['x_right'], band['y_bottom']],
                "conf": float(round(conf, 3))
            })
        return out

    # --- GT path: ALWAYS line-by-line ---
    # Decide arrows from text itself (not bbox)
    def arrow_from_text(t: str) -> str:
        s = t.lstrip()
        return "sub" if s.startswith(("-", "→", "↳")) else "main"

    # If bbox count matches GT, align 1:1 by order (top->bottom)
    out = []
    if len(bands) == len(gt_lines):
        for band, t in zip(bands, gt_lines):
            out.append({
                "arrow": arrow_from_text(t),
                "text": t,
                "bbox": [band['x_left'], band['y_top'], band['x_right'], band['y_bottom']],
                "conf": 1.0
            })
        return out

    # If mismatch: still output GT lines (correct line breaks), bbox=None
    # You can fix bbox alignment later without breaking training.
    for t in gt_lines:
        out.append({
            "arrow": arrow_from_text(t),
            "text": t,
            "bbox": None,
            "conf": 1.0
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Segment lines by page and reorganize JSON.")
    parser.add_argument('--input', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--debug-dir', default=None)
    parser.add_argument('--openai-api-key', default=os.environ.get('OPENAI_API_KEY'))
    parser.add_argument('--first-page-only', action='store_true', help='Process only the first page and print results')
    args = parser.parse_args()

    data = load_json(args.input)
    entries = data.get('entries', data if isinstance(data, list) else [])
    output = []
    missing_images = []
    for idx, entry in enumerate(entries):
        file_number = entry.get('file_number')
        if file_number is None:
            continue
        try:
            image_path = find_image_for_page(args.images_dir, int(file_number))
        except Exception as e:
            print(f"Missing image for file_number {file_number}: {e}")
            missing_images.append(file_number)
            continue
        gt_lines, gt_path = read_gt_txt(args.images_dir, int(file_number))
        lines = process_page(image_path, file_number=file_number, debug_dir=args.debug_dir, openai_api_key=args.openai_api_key, gt_lines=gt_lines)
        print(f"file_number {file_number}: {len(lines)} lines generated. Sample: {[l['text'] for l in lines[:3]]}")
        # If --first-page-only, print all lines and draw overlay, then exit
        if args.first_page_only:
            print("\n=== Detected lines for first page ===")
            for i, l in enumerate(lines):
                print(f"Line {i+1}: '{l['text']}' bbox={l['bbox']} conf={l['conf']}")
            # Draw overlay with only accepted lines
            import cv2
            from pathlib import Path
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            for l in lines:
                if l['bbox'] is None:
                    continue
                x0, y0, x1, y1 = l['bbox']
                cv2.rectangle(img, (x0, y0), (x1, y1), (0,255,0), 2)
                cv2.putText(img, l['text'][:30], (x0, max(y0-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            outdir = args.debug_dir or '.'
            Path(outdir).mkdir(parents=True, exist_ok=True)
            outpath = os.path.join(outdir, f"{file_number:03}_accepted_lines.jpg")
            cv2.imwrite(outpath, img)
            print(f"Overlay with accepted lines saved to {outpath}")
            return
        out_entry = dict(entry)
        out_entry['lines'] = lines
        output.append(out_entry)
        # Write incrementally every 5 entries or at the end
        if (idx+1) % 5 == 0 or (idx+1) == len(entries):
            save_json(output, args.output)
            print(f"[Incremental write] {len(output)} entries written to {args.output}")
        # Diagnostics: compare gt.txt line count to detected lines
        if gt_lines is not None:
            if abs(len(gt_lines) - len(lines)) > 2:
                print(f"Line count mismatch for file_number {file_number}: gt.txt has {len(gt_lines)}, detected {len(lines)} (gt: {gt_path})")
        if len(lines) < 3 or len(lines) > 80:
            print(f"Sanity check: file_number {file_number} has {len(lines)} lines (flag for review)")
    print(f"Wrote {len(output)} pages to {args.output}")
    if missing_images:
        print(f"Missing images for file_numbers: {missing_images}")

if __name__ == '__main__':
    main()
