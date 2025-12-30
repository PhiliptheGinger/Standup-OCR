#!/usr/bin/env python3
"""Ask GPT to place bounding boxes for known ground truth text.

Instead of OCR validation, we just tell GPT the exact text and ask it
to draw tight boxes around each line.
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

def place_boxes_for_text(image_path: Path, text_lines: list[str]) -> list[dict]:
    """Ask GPT to place bounding boxes around known text lines."""
    
    if OpenAI is None or Image is None:
        print("ERROR: Required libraries not installed")
        print("Install with: pip install openai Pillow")
        sys.exit(1)
    
    api_key = load_env_api_key()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Load image
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    original_width, original_height = img.size
    
    # Encode image
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    image_b64 = base64.standard_b64encode(buffer.getvalue()).decode('utf-8')
    
    # Build prompt
    text_list = "\n".join([f'{i+1}. "{line}"' for i, line in enumerate(text_lines)])
    
    prompt = f"""This is a handwritten page. Below are the exact text lines written on it.

Please find each line on the page and provide a bounding box (in pixels).

Text lines:
{text_list}

Return valid JSON array:
[
  {{"text": "line 1 text here", "bbox": [x1, y1, x2, y2]}},
  {{"text": "line 2 text here", "bbox": [x1, y1, x2, y2]}},
  ...
]

Use pixel coordinates (0,0 is top-left). Image is {original_width}×{original_height} pixels."""
    
    print(f"Asking GPT to place boxes for {len(text_lines)} lines...")
    print(f"Image size: {original_width}x{original_height}")
    
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
    
    # Parse JSON
    try:
        segments = json.loads(response_text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            segments = json.loads(json_match.group(0))
        else:
            print("ERROR: Could not parse GPT response as JSON")
            return []
    
    # Detect GPT's working coordinate space and rescale to original
    if segments:
        gpt_max_x = max(seg['bbox'][2] for seg in segments if 'bbox' in seg)
        gpt_max_y = max(seg['bbox'][3] for seg in segments if 'bbox' in seg)
        
        # If GPT's max coordinates are significantly smaller than image size,
        # it's working in a downsampled space
        if gpt_max_x < original_width * 0.8 or gpt_max_y < original_height * 0.8:
            # Estimate GPT's working dimensions (usually ~2048 max or similar)
            gpt_width = int(gpt_max_x * 1.2)  # Add 20% margin
            gpt_height = int(gpt_max_y * 1.2)
            
            scale_x = original_width / gpt_width
            scale_y = original_height / gpt_height
            
            print(f"Detected GPT working in ~{gpt_width}×{gpt_height} space")
            print(f"Rescaling to {original_width}×{original_height} (scale: {scale_x:.2f}x, {scale_y:.2f}y)")
            
            for seg in segments:
                if 'bbox' in seg:
                    bbox = seg['bbox']
                    seg['bbox'] = [
                        int(bbox[0] * scale_x),
                        int(bbox[1] * scale_y),
                        int(bbox[2] * scale_x),
                        int(bbox[3] * scale_y),
                    ]
        else:
            print("Coordinates appear to be in full image resolution")
    
    return segments

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ask GPT to place bounding boxes for known ground truth text.",
        epilog="""
Example:
  python scripts/place_gt_boxes.py data/train/images/001.jpg --text-file data/train/001.txt
        """,
    )
    
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--text-file", help="Path to text file with one line per text line")
    parser.add_argument("--text-lines", nargs='+', help="Text lines directly as arguments")
    parser.add_argument("--output-json", help="Save results to JSON (default: <image>.gpt_segments.json)")
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    
    # Load text lines
    if args.text_file:
        text_lines = Path(args.text_file).read_text(encoding='utf-8').strip().split('\n')
    elif args.text_lines:
        text_lines = args.text_lines
    else:
        print("ERROR: Must provide either --text-file or --text-lines")
        sys.exit(1)
    
    segments = place_boxes_for_text(image_path, text_lines)
    
    # Convert to standard format
    result = []
    for seg in segments:
        result.append({
            'text': seg.get('text', ''),
            'bbox': seg.get('bbox', [0, 0, 0, 0])
        })
    
    print(f"\nPlaced {len(result)} boxes")
    
    # Save
    output_path = Path(args.output_json) if args.output_json else image_path.parent / f"{image_path.stem}.gpt_segments.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved to {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
