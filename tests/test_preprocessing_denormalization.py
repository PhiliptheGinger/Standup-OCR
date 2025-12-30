"""Tests for coordinate denormalization in preprocessing."""
from __future__ import annotations

from pathlib import Path
import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    denormalize_coordinates,
    OrientationResult,
    OrientationOptions,
    normalize_page_orientation,
)
from PIL import Image
import numpy as np


def test_denormalize_coordinates_identity_no_rotation():
    """Test that denormalization returns original coordinates when no rotation applied."""
    coords = [(10.0, 20.0), (100.0, 200.0)]
    result = OrientationResult(applied=False)
    
    denorm = denormalize_coordinates(
        coords,
        original_size=(200, 300),
        normalized_size=(200, 300),
        rotation_meta=result,
    )
    
    assert denorm == coords


def test_denormalize_coordinates_empty_applied():
    """Test that empty coords returns empty list regardless of applied flag."""
    coords: list[tuple[float, float]] = []
    result = OrientationResult(applied=True, rotated_quadrants=1)
    
    denorm = denormalize_coordinates(
        coords,
        original_size=(200, 300),
        normalized_size=(300, 200),
        rotation_meta=result,
    )
    
    assert denorm == []


def test_denormalize_coordinates_clamps_to_bounds():
    """Test that denormalized coordinates are clamped to original image bounds."""
    coords = [(-10.0, -10.0), (1000.0, 1000.0)]
    
    result = OrientationResult(applied=True, rotated_quadrants=0)
    
    denorm = denormalize_coordinates(
        coords,
        original_size=(200, 300),
        normalized_size=(200, 300),
        rotation_meta=result,
    )
    
    # Should be clamped to [0, 199] x [0, 299]
    assert denorm[0] == (0.0, 0.0)
    assert denorm[1] == (199.0, 299.0)


def test_denormalize_coordinates_landscape_rotation_simple():
    """Test denormalization of a simple 90° CCW landscape rotation.
    
    Original: 200x300 (portrait)
    After 90° CCW: 300x200 (landscape)
    
    Point (x, y) in portrait -> rotated 90° CCW -> (y, 200-1-x)
    To reverse: (x_rot, y_rot) -> (y_rot, 300-1-x_rot)
    """
    # Original portrait image
    original_size = (200, 300)
    # After 90° CCW rotation, dimensions swap: 300x200
    normalized_size = (300, 200)
    
    # In normalized (rotated) space, a point at (50, 100)
    coords = [(50.0, 100.0)]
    
    result = OrientationResult(
        applied=True,
        rotated_quadrants=1,
        flipped=False,
        width=normalized_size[0],
        height=normalized_size[1],
    )
    
    denorm = denormalize_coordinates(
        coords,
        original_size,
        normalized_size,
        result,
    )
    
    # Reverse 90° CCW: (x, y) -> (H-1-y, x) where H=200 (normalized height)
    # (50, 100) -> (200-1-100, 50) = (99, 50)
    assert len(denorm) == 1
    x, y = denorm[0]
    assert 95 <= x <= 105  # Should be around 99
    assert 45 <= y <= 55   # Should be around 50


def test_denormalize_with_full_normalization_pipeline():
    """Test denormalization with a real portrait image that gets normalized.
    
    Create a portrait image, normalize it with force_landscape=True,
    then verify denormalization logic with simulated coordinates.
    """
    # Create a simple portrait image: 100x200 (tall)
    array = np.ones((200, 100, 3), dtype=np.uint8) * 200
    # Add some ink in top half to prevent upright flip
    array[10:30, 10:90] = 50
    
    image = Image.fromarray(array, "RGB")
    
    options = OrientationOptions(
        max_skew=0,
        force_landscape=True,
        force_upright=True,
    )
    
    normalized, meta = normalize_page_orientation(image, options=options)
    
    # Original: 100x200 (portrait), after normalization should be landscape (200x100)
    assert meta.applied
    # After 90° CCW rotation: width and height get swapped in the output
    assert normalized.width == 200  # swapped from 100
    assert normalized.height == 100  # swapped from 200
    
    # Create a bbox in normalized space (on the 200x100 image)
    # E.g., top-left at (20, 30)
    normalized_coords = [(20.0, 30.0)]
    
    denorm = denormalize_coordinates(
        normalized_coords,
        original_size=(100, 200),  # original portrait size
        normalized_size=(200, 100),  # landscape normalized size
        rotation_meta=meta,
    )
    
    # Should map back to original image space (100x200)
    assert len(denorm) == 1
    x, y = denorm[0]
    # Check bounds
    assert 0 <= x < 100
    assert 0 <= y < 200
