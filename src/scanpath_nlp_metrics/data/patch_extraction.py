"""
Image patch extraction utilities.
"""

from __future__ import annotations

from typing import Tuple

from PIL import Image


def extract_patch(
    image: Image.Image,
    x: float,
    y: float,
    patch_size: int,
) -> Image.Image:
    """Extract a square patch centered at (x, y) from an image."""
    W, H = image.size
    half = patch_size // 2

    left = max(0, int(x) - half)
    upper = max(0, int(y) - half)
    right = min(W, int(x) + half)
    lower = min(H, int(y) + half)

    if right - left < patch_size:
        left = max(0, right - patch_size)
    if lower - upper < patch_size:
        upper = max(0, lower - patch_size)

    return image.crop((left, upper, right, lower))


def extract_patch_with_padding(
    image: Image.Image,
    x: float,
    y: float,
    patch_size: int,
    padding_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Extract a square patch with padding if needed."""
    W, H = image.size
    half = patch_size // 2

    left = int(x) - half
    upper = int(y) - half
    right = int(x) + half
    lower = int(y) + half

    patch = Image.new("RGB", (patch_size, patch_size), padding_color)

    src_left = max(0, left)
    src_upper = max(0, upper)
    src_right = min(W, right)
    src_lower = min(H, lower)

    dst_left = max(0, -left)
    dst_upper = max(0, -upper)
    dst_right = dst_left + (src_right - src_left)
    dst_lower = dst_upper + (src_lower - src_upper)

    if src_right > src_left and src_lower > src_upper:
        cropped = image.crop((src_left, src_upper, src_right, src_lower))
        patch.paste(cropped, (dst_left, dst_upper))

    return patch
