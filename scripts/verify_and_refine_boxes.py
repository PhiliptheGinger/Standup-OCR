#!/usr/bin/env python3
"""Refine boxes by verifying GPT can read the correct text from each crop.

This script crops each box, asks GPT what text it sees, and adjusts boxes
where GPT reads something different from the ground truth.
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
    from PIL import Image, ImageDraw, ImageOps
except ImportError:
    Image = None

def load_env_api_key() -> Optional[str]:
    """Load OpenAI API key from .env or environment."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    import os
    return os.getenv("OPENAI_API_KEY")

def ocr_crop_with_gpt(client: OpenAI, crop: Image.Image) -> str:
    """Ask GPT what handwritten text is in this crop."""
    buffer = BytesIO()
    crop.save(buffer, format='PNG')
    crop_b64 = base64.standard_b64encode(buffer.getvalue()).decode('utf-8')
    
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{crop_b64}"}},
                    {"type": "text", "text": "What handwritten text is in this image? Return ONLY the text, nothing else."},
                ],
            }
        ],
    )
    
    return response.choices[0].message.content.strip()

def draw_annotated_image(img: Image.Image, segments: list[dict], gt_lines: list[str], matches: list[bool]) -> bytes:
    """Draw boxes colored by verification status."""
    draw = ImageDraw.Draw(img)
    
    for idx, (seg, gt, match) in enumerate(zip(segments, gt_lines, matches), 1):
        bbox = seg['bbox']
        color = 'green' if match else 'red'
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=color, width=8)
        draw.text((bbox[0], bbox[1] - 40), f"{idx}", fill=color)
    
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

def refine_with_crop_verification(image_path: Path, segments: list[dict], gt_lines: list[str], iteration: int) -> tuple[list[dict], bool]:
    """Refine boxes by verifying crops match ground truth."""
    
    if OpenAI is None or Image is None:
        print("ERROR: Missing dependencies")
        sys.exit(1)
    
    api_key = load_env_api_key()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img_width, img_height = img.size
    
    print(f"\nIteration {iteration}: Verifying {len(segments)} crops with GPT OCR...")
    
    # Verify each crop
    matches = []
    mismatches = []
    
    for idx, (seg, gt) in enumerate(zip(segments, gt_lines), 1):
        bbox = seg['bbox']
        crop = img.crop(bbox)
        
        # Ask GPT what text is in the crop
        ocr_text = ocr_crop_with_gpt(client, crop)
        
        # Normalize for comparison
        gt_norm = gt.lower().strip()
        ocr_norm = ocr_text.lower().strip()
        
        match = gt_norm == ocr_norm
        matches.append(match)
        
        status = "✓" if match else "✗"
        print(f"  {idx}. {status} GT: '{gt}' | OCR: '{ocr_text}'")
        
        if not match:
            mismatches.append({
                'line': idx,
                'bbox': bbox,
                'gt': gt,
                'ocr': ocr_text
            })
    
    if not mismatches:
        print("\n✓ All boxes verified - converged!")
        return segments, True
    
    print(f"\n✗ {len(mismatches)}/{len(segments)} boxes need adjustment")
    
    # Draw annotated image
    annotated = draw_annotated_image(img, segments, gt_lines, matches)
    image_b64 = base64.standard_b64encode(annotated).decode('utf-8')
    
    # Build mismatch report
    mismatch_report = "\n".join([
        f"Line {m['line']}: bbox={m['bbox']}\n  Should contain: \"{m['gt']}\"\n  Currently reads as: \"{m['ocr']}\""
        for m in mismatches
    ])
    
    prompt = f"""I've drawn boxes on this handwritten page with color coding:
- GREEN: When I crop and OCR that box, it correctly reads the expected text
- RED: When I crop and OCR that box, it reads WRONG text

Mismatches to fix:
{mismatch_report}

For each RED box, adjust the bounding box so that when cropped, it will contain ONLY the intended text.

Return ONLY valid JSON array with adjustments:
[
  {{"line": 1, "bbox": [x1, y1, x2, y2], "reason": "explanation"}},
  ...
]

Image is {img_width}×{img_height} pixels. Be precise with coordinates."""
    
    print("Asking GPT to fix mismatched boxes...")
    
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    
    response_text = response.choices[0].message.content
    print(f"\nGPT Response:\n{response_text}\n")
    
    try:
        adjustments = json.loads(response_text)
    except:
        import re
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if match:
            adjustments = json.loads(match.group(0))
        else:
            print("No valid JSON found")
            return segments, False
    
    # Detect and rescale if needed
    if adjustments:
        gpt_max_x = max(adj['bbox'][2] for adj in adjustments if 'bbox' in adj)
        gpt_max_y = max(adj['bbox'][3] for adj in adjustments if 'bbox' in adj)
        
        if gpt_max_x < img_width * 0.8 or gpt_max_y < img_height * 0.8:
            gpt_width = int(gpt_max_x * 1.2)
            gpt_height = int(gpt_max_y * 1.2)
            scale_x = img_width / gpt_width
            scale_y = img_height / gpt_height
            
            print(f"Rescaling GPT coords from ~{gpt_width}×{gpt_height} to {img_width}×{img_height}")
            
            for adj in adjustments:
                if 'bbox' in adj:
                    bbox = adj['bbox']
                    adj['bbox'] = [
                        int(bbox[0] * scale_x),
                        int(bbox[1] * scale_y),
                        int(bbox[2] * scale_x),
                        int(bbox[3] * scale_y),
                    ]
    
    print(f"Applying {len(adjustments)} adjustments...")
    refined = [s.copy() for s in segments]
    
    for adj in adjustments:
        line_num = adj.get('line')
        new_bbox = adj.get('bbox')
        reason = adj.get('reason', '')
        
        if line_num and new_bbox and 1 <= line_num <= len(refined):
            old = refined[line_num - 1]['bbox']
            refined[line_num - 1]['bbox'] = new_bbox
            print(f"  Line {line_num}: {old} -> {new_bbox} ({reason})")
    
    return refined, False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Refine boxes using GPT crop verification.")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--gt-text", help="Ground truth text file (default: <image>.gt.txt)")
    parser.add_argument("--segments-json", help="Segments JSON (default: <image>.gpt_segments.json)")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max iterations (default: 5)")
    parser.add_argument("--output-json", help="Output JSON (default: overwrites input)")
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    
    # Auto-detect paths
    gt_path = Path(args.gt_text) if args.gt_text else image_path.parent / f"{image_path.stem}.gt.txt"
    if not gt_path.exists():
        print(f"ERROR: Ground truth not found: {gt_path}")
        sys.exit(1)
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_lines = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(gt_lines)} GT lines from {gt_path}")
    
    segments_json = Path(args.segments_json) if args.segments_json else image_path.parent / f"{image_path.stem}.gpt_segments.json"
    if not segments_json.exists():
        print(f"ERROR: Segments not found: {segments_json}")
        sys.exit(1)
    
    with open(segments_json, 'r') as f:
        segments = json.load(f)
    
    print(f"Loaded {len(segments)} segments\n")
    
    # Refine iteratively
    for iteration in range(1, args.max_iterations + 1):
        segments, converged = refine_with_crop_verification(image_path, segments, gt_lines, iteration)
        if converged:
            print(f"\n✓ Converged after {iteration} iterations")
            break
    else:
        print(f"\n⚠ Reached max {args.max_iterations} iterations")
    
    # Save
    output = Path(args.output_json) if args.output_json else segments_json
    output.write_text(json.dumps(segments, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {output}")

if __name__ == "__main__":
    sys.exit(main())
