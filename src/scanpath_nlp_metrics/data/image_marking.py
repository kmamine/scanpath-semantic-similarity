"""
Image marking utilities for fixation visualization.
"""

from __future__ import annotations

from typing import Tuple, List

from PIL import Image, ImageDraw


def draw_fixation_marker(
    image: Image.Image,
    x: float,
    y: float,
    radius: int = 100,
    outline_color: Tuple[int, int, int] = (255, 0, 0),
    outline_width: int = 3,
    dot_radius: int = 5,
    dot_color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Draw a circle outline with center dot at a fixation location."""
    img_copy = image.copy()

    if img_copy.mode != "RGB":
        img_copy = img_copy.convert("RGB")

    draw = ImageDraw.Draw(img_copy)

    cx, cy = int(x), int(y)

    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=outline_color,
        width=outline_width,
    )

    draw.ellipse(
        [cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius],
        fill=dot_color,
        outline=dot_color,
    )

    return img_copy
