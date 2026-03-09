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


def draw_fixation_marker_with_label(
    image: Image.Image,
    x: float,
    y: float,
    label: str,
    radius: int = 100,
    outline_color: Tuple[int, int, int] = (255, 0, 0),
    outline_width: int = 3,
    dot_radius: int = 5,
    dot_color: Tuple[int, int, int] = (255, 0, 0),
    label_color: Tuple[int, int, int] = (255, 255, 255),
    label_bg: Tuple[int, int, int] = (255, 0, 0),
    font_size: int = 20,
) -> Image.Image:
    """Draw a circle with center dot and a label at the fixation location."""
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

    text_bbox = draw.textbbox((0, 0), label)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    label_x = cx + radius + 10
    label_y = cy - text_h // 2

    if label_x + text_w > image.width:
        label_x = cx - radius - text_w - 10

    padding = 4
    draw.rectangle(
        [
            label_x - padding,
            label_y - padding,
            label_x + text_w + padding,
            label_y + text_h + padding,
        ],
        fill=label_bg,
    )
    draw.text((label_x, label_y), label, fill=label_color)

    return img_copy


def draw_all_fixations(
    image: Image.Image,
    fixations: List[Tuple[float, float]],
    radius: int = 100,
    outline_color: Tuple[int, int, int] = (255, 0, 0),
    outline_width: int = 2,
    dot_radius: int = 4,
    dot_color: Tuple[int, int, int] = (255, 0, 0),
    show_numbers: bool = True,
    number_color: Tuple[int, int, int] = (255, 255, 255),
    number_bg: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Draw all fixation markers on a single image."""
    img_copy = image.copy()

    if img_copy.mode != "RGB":
        img_copy = img_copy.convert("RGB")

    draw = ImageDraw.Draw(img_copy)

    for i, (x, y) in enumerate(fixations):
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

        if show_numbers:
            label = str(i + 1)
            text_bbox = draw.textbbox((0, 0), label)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            label_x = cx - text_w // 2
            label_y = cy - text_h // 2

            padding = 2
            draw.rectangle(
                [
                    label_x - padding,
                    label_y - padding,
                    label_x + text_w + padding,
                    label_y + text_h + padding,
                ],
                fill=number_bg,
            )
            draw.text((label_x, label_y), label, fill=number_color)

    return img_copy


def highlight_current_fixation(
    image: Image.Image,
    fixations: List[Tuple[float, float]],
    current_idx: int,
    radius: int = 100,
    outline_color: Tuple[int, int, int] = (255, 0, 0),
    outline_width: int = 3,
    dot_radius: int = 5,
    dot_color: Tuple[int, int, int] = (255, 0, 0),
    other_outline_color: Tuple[int, int, int] = (128, 128, 128),
    other_outline_width: int = 1,
    other_dot_radius: int = 3,
    other_dot_color: Tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """Draw all fixations but highlight the current one differently."""
    img_copy = image.copy()

    if img_copy.mode != "RGB":
        img_copy = img_copy.convert("RGB")

    draw = ImageDraw.Draw(img_copy)

    for i, (x, y) in enumerate(fixations):
        cx, cy = int(x), int(y)

        if i == current_idx:
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
        else:
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                outline=other_outline_color,
                width=other_outline_width,
            )
            draw.ellipse(
                [
                    cx - other_dot_radius,
                    cy - other_dot_radius,
                    cx + other_dot_radius,
                    cy + other_dot_radius,
                ],
                fill=other_dot_color,
                outline=other_dot_color,
            )

    return img_copy
