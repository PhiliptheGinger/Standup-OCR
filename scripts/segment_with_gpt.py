#!/usr/bin/env python3
"""Segment handwritten pages using GPT-4 vision directly.

This script asks GPT to identify text lines and return bounding boxes as JSON,
bypassing Kraken segmentation entirely.
"""

import json
import sys
import base64
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def load_env_api_key() -> Optional[str]:
    """Load OpenAI API key from .env or environment."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    import os
    return os.getenv("OPENAI_API_KEY")

def segment_with_gpt(image_path: Path) -> list[dict]:
    """Ask GPT to identify text lines and return bounding boxes.
    
    Returns:
        List of dicts with keys: 'text', 'bbox' [x1, y1, x2, y2]
    """
    
    if OpenAI is None:
        print("ERROR: OpenAI library not installed. Install with: pip install openai")
        sys.exit(1)
    
    api_key = load_env_api_key()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment or .env file")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)
    
    # Load image to get dimensions for rescaling
    try:
        from PIL import Image
        pil_img = Image.open(image_path)
        original_width, original_height = pil_img.size
    except Exception as e:
        print(f"ERROR: Could not load image: {e}")
        sys.exit(1)
    
    # Load and encode image
    with open(image_path, 'rb') as f:
        image_b64 = base64.standard_b64encode(f.read()).decode('utf-8')
    
    # Determine MIME type
    suffix = image_path.suffix.lower()
    mime_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }.get(suffix, 'image/jpeg')
    
    print(f"Sending {image_path.name} to GPT-4o for segmentation...")
    print(f"Original image size: {original_width}x{original_height}")
    
    prompt = """Identify all distinct lines of handwritten text on this page. 
For each line, provide:
1. The text content
2. Its bounding box as [x1, y1, x2, y2] in pixel coordinates (top-left to bottom-right)

Return ONLY valid JSON array, no other text:
[
  {"text": "line content here", "bbox": [x1, y1, x2, y2]},
  ...
]

Be precise with coordinates. Include all visible text, even if partially legible."""
    
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
                            "url": f"data:{mime_type};base64,{image_b64}",
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
        lines = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            lines = json.loads(json_match.group(0))
        else:
            print("ERROR: Could not parse GPT response as JSON")
            return []
    
    # Rescale coordinates to original image dimensions
    # GPT works in a scaled coordinate space (typically ~512x384)
    if lines:
        # Estimate GPT's working dimensions from max bbox values
        gpt_max_x = max(line['bbox'][2] for line in lines if 'bbox' in line)
        gpt_max_y = max(line['bbox'][3] for line in lines if 'bbox' in line)
        
        # Common GPT scaling: it uses ~512x384 for analysis
        # But we can estimate from the actual max values
        gpt_width = max(512, int(gpt_max_x * 1.2))
        gpt_height = max(384, int(gpt_max_y * 1.2))
        
        scale_x = original_width / gpt_width
        scale_y = original_height / gpt_height
        
        print(f"Detected GPT working space: ~{gpt_width}x{gpt_height}")
        print(f"Rescaling to original: {original_width}x{original_height} (scale: {scale_x:.2f}x, {scale_y:.2f}y)")
        
        for line in lines:
            if 'bbox' in line:
                bbox = line['bbox']
                line['bbox'] = [
                    int(bbox[0] * scale_x),
                    int(bbox[1] * scale_y),
                    int(bbox[2] * scale_x),
                    int(bbox[3] * scale_y),
                ]
    
    return lines

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Segment handwritten pages using GPT-4o vision (bypasses Kraken).",
        epilog="""
Example:
  python scripts/segment_with_gpt.py data/train/images/001.jpg
        """,
    )
    
    parser.add_argument("image", help="Path to image to segment")
    parser.add_argument("--output-json", help="Save results to JSON file (default: <image_stem>.gpt_segments.json)")
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    segments = segment_with_gpt(image_path)
    
    print(f"\nDetected {len(segments)} text lines")
    
    if segments and args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps(segments, indent=2) + "\n", encoding="utf-8")
        print(f"Saved to {output_path}")
    elif segments:
        output_path = image_path.parent / f"{image_path.stem}.gpt_segments.json"
        output_path.write_text(json.dumps(segments, indent=2) + "\n", encoding="utf-8")
        print(f"Saved to {output_path}")
    
    return 0 if segments else 1

if __name__ == "__main__":
    sys.exit(main())
