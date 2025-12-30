#!/usr/bin/env python3
"""Iteratively refine GPT bounding boxes using visual feedback.

This script shows GPT its current box placements and asks it to adjust
them to better fit the actual text boundaries.
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
    from PIL import Image, ImageDraw, ImageFont
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

def draw_boxes_on_image(image_path: Path, segments: list[dict]) -> bytes:
    """Draw current boxes on image and return as PNG bytes."""
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Draw boxes with line numbers
    for idx, seg in enumerate(segments, 1):
        bbox = seg['bbox']
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline='red', width=8)
        # Add line number
        draw.text((bbox[0], bbox[1] - 30), f"{idx}", fill='red')
    
    # Convert to PNG bytes
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

def refine_boxes_with_gpt(image_path: Path, segments: list[dict], iteration: int = 1) -> list[dict]:
    """Ask GPT to refine bounding box coordinates based on visual feedback."""
    
    if OpenAI is None:
        print("ERROR: OpenAI library not installed. Install with: pip install openai")
        sys.exit(1)
    
    if Image is None:
        print("ERROR: Pillow library not installed. Install with: pip install Pillow")
        sys.exit(1)
    
    api_key = load_env_api_key()
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment or .env file")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Draw current boxes on image
    print(f"Iteration {iteration}: Drawing current boxes on image...")
    annotated_image = draw_boxes_on_image(image_path, segments)
    image_b64 = base64.standard_b64encode(annotated_image).decode('utf-8')
    
    # Build prompt with current segments
    segments_text = "\n".join([
        f'{idx}. "{seg["text"]}" -> bbox: {seg["bbox"]}'
        for idx, seg in enumerate(segments, 1)
    ])
    
    prompt = f"""I've drawn red bounding boxes on this handwritten page (numbered 1-{len(segments)}).
Each box should tightly fit its corresponding text line.

Current boxes and text:
{segments_text}

Please analyze the fit and provide ONLY a JSON array with adjustments needed.
For each line that needs adjustment, return:
{{"line": <number>, "bbox": [x1, y1, x2, y2], "reason": "brief explanation"}}

If a box looks good, omit it from the response.
Return ONLY valid JSON array, no other text:
[
  {{"line": 1, "bbox": [x1, y1, x2, y2], "reason": "move right 10px"}},
  ...
]

Be precise with pixel coordinates. Focus on making boxes tightly fit the text."""
    
    print(f"Sending annotated image to GPT-4o for refinement...")
    
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
    
    # Parse JSON adjustments
    try:
        adjustments = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            adjustments = json.loads(json_match.group(0))
        else:
            print("WARNING: Could not parse GPT response as JSON, no adjustments made")
            return segments
    
    if not adjustments:
        print("No adjustments needed!")
        return segments
    
    # Apply adjustments
    refined_segments = segments.copy()
    for adj in adjustments:
        line_num = adj.get('line')
        new_bbox = adj.get('bbox')
        reason = adj.get('reason', 'no reason given')
        
        if line_num and new_bbox and 1 <= line_num <= len(refined_segments):
            old_bbox = refined_segments[line_num - 1]['bbox']
            refined_segments[line_num - 1]['bbox'] = new_bbox
            print(f"Line {line_num}: {old_bbox} -> {new_bbox} ({reason})")
    
    return refined_segments

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Iteratively refine GPT bounding boxes using visual feedback.",
        epilog="""
Example:
  python scripts/refine_gpt_boxes.py data/train/images/001.jpg --iterations 3
        """,
    )
    
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--segments-json", help="Path to GPT segments JSON (default: auto-detect)")
    parser.add_argument("--iterations", type=int, default=2, help="Number of refinement iterations (default: 2)")
    parser.add_argument("--output-json", help="Save refined results to JSON file (default: overwrites input)")
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    
    # Auto-detect segments JSON
    segments_json = Path(args.segments_json) if args.segments_json else image_path.parent / f"{image_path.stem}.gpt_segments.json"
    
    if not segments_json.exists():
        print(f"ERROR: Segments JSON not found: {segments_json}")
        print("Run segment_with_gpt.py first to generate initial segments")
        sys.exit(1)
    
    # Load initial segments
    with open(segments_json, 'r') as f:
        segments = json.load(f)
    
    print(f"Loaded {len(segments)} initial segments from {segments_json}")
    
    # Iterative refinement
    for iteration in range(1, args.iterations + 1):
        segments = refine_boxes_with_gpt(image_path, segments, iteration)
        print(f"\nIteration {iteration} complete\n")
    
    # Save refined segments
    output_path = Path(args.output_json) if args.output_json else segments_json
    output_path.write_text(json.dumps(segments, indent=2) + "\n", encoding="utf-8")
    print(f"Saved refined segments to {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
