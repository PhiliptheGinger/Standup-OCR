#!/usr/bin/env python3
"""Refine GPT boxes using ground truth text validation.

This script crops each box, OCRs it, and asks GPT to adjust boxes
where the OCR doesn't match the expected ground truth text.
"""

import json
import sys
import base64
from pathlib import Path
from io import BytesIO
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from PIL import Image, ImageDraw
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None

def load_env_api_key() -> Optional[str]:
    """Load OpenAI API key from .env or environment."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    import os
    return os.getenv("OPENAI_API_KEY")

def ocr_crop(img: Image.Image, bbox: list) -> str:
    """OCR a cropped region of the image."""
    crop = img.crop(bbox)
    try:
        text = pytesseract.image_to_string(crop, config='--psm 7').strip()
        return text
    except:
        return ""

def draw_boxes_with_errors(img: Image.Image, segments: list[dict], ocr_results: list[str]) -> bytes:
    """Draw boxes colored by OCR match status."""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for idx, seg in enumerate(segments):
        bbox = seg['bbox']
        gt_text = seg['text']
        ocr_text = ocr_results[idx]
        
        # Green if match, red if mismatch
        color = 'green' if ocr_text.lower().strip() == gt_text.lower().strip() else 'red'
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=color, width=8)
        draw.text((bbox[0], bbox[1] - 40), f"{idx+1}", fill=color)
    
    buffer = BytesIO()
    img_copy.save(buffer, format='PNG')
    return buffer.getvalue()

def refine_with_ground_truth(image_path: Path, segments: list[dict], gt_lines: list[str], iteration: int) -> tuple[list[dict], bool]:
    """Refine boxes using ground truth text validation."""
    
    if OpenAI is None or Image is None or pytesseract is None:
        print("ERROR: Required libraries not installed")
        print("Install with: pip install openai Pillow pytesseract")
        sys.exit(1)
    
    api_key = load_env_api_key()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Load image
    img = Image.open(image_path)
    
    # OCR each crop
    print(f"\nIteration {iteration}: OCR'ing {len(segments)} crops...")
    ocr_results = []
    mismatches = []
    
    for idx, seg in enumerate(segments, 1):
        bbox = seg['bbox']
        gt_text = seg['text']
        
        ocr_text = ocr_crop(img, bbox)
        ocr_results.append(ocr_text)
        
        match = ocr_text.lower().strip() == gt_text.lower().strip()
        status = "✓" if match else "✗"
        print(f"  {idx}. {status} GT: '{gt_text}' | OCR: '{ocr_text}'")
        
        if not match:
            mismatches.append({
                'line': idx,
                'bbox': bbox,
                'gt': gt_text,
                'ocr': ocr_text
            })
    
    if not mismatches:
        print("\n✓ All boxes match ground truth!")
        return segments, True
    
    print(f"\n✗ {len(mismatches)}/{len(segments)} boxes need adjustment")
    
    # Draw annotated image
    annotated_image = draw_boxes_with_errors(img, segments, ocr_results)
    image_b64 = base64.standard_b64encode(annotated_image).decode('utf-8')
    
    # Build detailed mismatch report
    mismatch_report = "\n".join([
        f"Line {m['line']}: bbox={m['bbox']}\n  Expected: \"{m['gt']}\"\n  OCR got: \"{m['ocr']}\""
        for m in mismatches
    ])
    
    prompt = f"""I've drawn bounding boxes on this handwritten page with color coding:
- GREEN boxes: OCR output matches expected text
- RED boxes: OCR output doesn't match expected text

Mismatches that need fixing:
{mismatch_report}

For each RED box, adjust the bounding box coordinates to better capture the intended text.
The boxes should tightly fit the text boundaries to improve OCR accuracy.

Return ONLY a JSON array with adjustments for MISMATCHED lines:
[
  {{"line": <number>, "bbox": [x1, y1, x2, y2], "reason": "brief explanation"}},
  ...
]

Be precise. Only adjust boxes that have OCR mismatches."""
    
    print(f"Asking GPT to fix {len(mismatches)} mismatched boxes...")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    )
    
    response_text = response.choices[0].message.content
    print(f"\nGPT Response:\n{response_text}\n")
    
    # Parse adjustments
    try:
        adjustments = json.loads(response_text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            adjustments = json.loads(json_match.group(0))
        else:
            print("WARNING: Could not parse GPT response")
            return segments, False
    
    # Apply adjustments
    refined_segments = [s.copy() for s in segments]
    for adj in adjustments:
        line_num = adj.get('line')
        new_bbox = adj.get('bbox')
        reason = adj.get('reason', '')
        
        if line_num and new_bbox and 1 <= line_num <= len(refined_segments):
            old_bbox = refined_segments[line_num - 1]['bbox']
            refined_segments[line_num - 1]['bbox'] = new_bbox
            print(f"Adjusted line {line_num}: {old_bbox} -> {new_bbox} ({reason})")
    
    return refined_segments, False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Refine GPT boxes using ground truth text validation.",
        epilog="""
Example:
  python scripts/refine_gpt_with_gt.py data/train/images/001.jpg --max-iterations 5
        """,
    )
    
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--gt-text", help="Ground truth text file (default: auto-detect <image>.gt.txt)")
    parser.add_argument("--segments-json", help="Path to segments JSON (default: auto-detect)")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max refinement iterations (default: 5)")
    parser.add_argument("--output-json", help="Save refined results (default: overwrites input)")
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    
    # Auto-detect GT text
    gt_text_path = Path(args.gt_text) if args.gt_text else image_path.parent / f"{image_path.stem}.gt.txt"
    if not gt_text_path.exists():
        print(f"ERROR: Ground truth not found: {gt_text_path}")
        sys.exit(1)
    
    with open(gt_text_path, 'r', encoding='utf-8') as f:
        gt_lines = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(gt_lines)} ground truth lines from {gt_text_path}")
    
    segments_json = Path(args.segments_json) if args.segments_json else image_path.parent / f"{image_path.stem}.gpt_segments.json"
    
    if not segments_json.exists():
        print(f"ERROR: Segments JSON not found: {segments_json}")
        sys.exit(1)
    
    with open(segments_json, 'r') as f:
        segments = json.load(f)
    
    print(f"Loaded {len(segments)} segments from {segments_json}")
    
    # Iterative refinement
    for iteration in range(1, args.max_iterations + 1):
        segments, converged = refine_with_ground_truth(image_path, segments, gt_lines, iteration)
        
        if converged:
            print(f"\n✓ Converged after {iteration} iterations")
            break
    else:
        print(f"\n⚠ Reached max iterations ({args.max_iterations}) without full convergence")
    
    # Save
    output_path = Path(args.output_json) if args.output_json else segments_json
    output_path.write_text(json.dumps(segments, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved refined segments to {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
