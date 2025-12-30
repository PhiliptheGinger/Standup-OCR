#!/usr/bin/env python3
"""Refine GPT boxes using ground truth text (no OCR needed)."""

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

def draw_boxes_with_labels(img: Image.Image, segments: list[dict], gt_lines: list[str]) -> bytes:
    """Draw boxes with GT text labels."""
    draw = ImageDraw.Draw(img)
    
    for idx, (seg, gt_text) in enumerate(zip(segments, gt_lines), 1):
        bbox = seg['bbox']
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline='red', width=8)
        # Add line number and GT text
        label = f"{idx}: {gt_text[:30]}"
        draw.text((bbox[0], bbox[1] - 40), label, fill='red')
    
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

def refine_with_gt(image_path: Path, segments: list[dict], gt_lines: list[str], iteration: int) -> tuple[list[dict], bool]:
    """Refine boxes using GT text."""
    
    if OpenAI is None or Image is None:
        print("ERROR: Missing dependencies")
        sys.exit(1)
    
    api_key = load_env_api_key()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    img = Image.open(image_path)
    
    print(f"\nIteration {iteration}: Drawing boxes with GT labels...")
    annotated = draw_boxes_with_labels(img, segments, gt_lines)
    image_b64 = base64.standard_b64encode(annotated).decode('utf-8')
    
    # Build prompt
    lines_text = "\n".join([
        f'{idx}. GT: "{gt}" | Current bbox: {seg["bbox"]}'
        for idx, (seg, gt) in enumerate(zip(segments, gt_lines), 1)
    ])
    
    prompt = f"""I've labeled this handwritten page with red boxes and the GROUND TRUTH text for each line.

Current boxes and their ground truth text:
{lines_text}

Please analyze if each box tightly fits its ground truth handwriting on the page.
Return ONLY valid JSON array with adjustments:
[
  {{"line": 1, "bbox": [x1, y1, x2, y2], "reason": "needs to move left"}},
  ...
]

Omit lines that look good. Be precise with pixel coordinates."""
    
    print("Asking GPT to refine boxes...")
    
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
    
    if not adjustments:
        print("✓ No adjustments needed - converged!")
        return segments, True
    
    # Detect if GPT is working in downsampled coordinate space and rescale
    img_width, img_height = img.size
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
    
    parser = argparse.ArgumentParser(description="Refine GPT boxes using ground truth text.")
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
    
    # Match segment count to GT count
    if len(segments) < len(gt_lines):
        print(f"WARNING: {len(segments)} segments but {len(gt_lines)} GT lines - may need initial segmentation")
    
    print(f"Loaded {len(segments)} segments\n")
    
    # Refine iteratively
    for iteration in range(1, args.max_iterations + 1):
        segments, converged = refine_with_gt(image_path, segments, gt_lines, iteration)
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
