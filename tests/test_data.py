"""
Unit tests for data utilities.
"""

import pytest
import numpy as np
from PIL import Image
from io import BytesIO
from scanpath_nlp_metrics.data import (
    extract_patch,
    extract_patch_with_padding,
    draw_fixation_marker,
)


class TestExtractPatch:
    def test_extract_center(self):
        img = Image.new("RGB", (500, 500), color=(255, 255, 255))
        patch = extract_patch(img, 250, 250, 96)

        assert patch.size[0] <= 96
        assert patch.size[1] <= 96

    def test_extract_near_edge(self):
        img = Image.new("RGB", (500, 500), color=(255, 255, 255))
        patch = extract_patch(img, 10, 10, 96)

        assert patch.size[0] > 0
        assert patch.size[1] > 0

    def test_extract_outside_bounds(self):
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        patch = extract_patch(img, -10, -10, 50)

        assert patch.size[0] > 0
        assert patch.size[1] > 0


class TestExtractPatchWithPadding:
    def test_exact_size(self):
        img = Image.new("RGB", (500, 500), color=(255, 255, 255))
        patch = extract_patch_with_padding(img, 250, 250, 96)

        assert patch.size == (96, 96)

    def test_corner_with_padding(self):
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        patch = extract_patch_with_padding(img, 0, 0, 96, padding_color=(0, 0, 0))

        assert patch.size == (96, 96)


class TestDrawFixationMarker:
    def test_draw_marker(self):
        img = Image.new("RGB", (500, 500), color=(255, 255, 255))
        marked = draw_fixation_marker(img, 250, 250, radius=50)

        assert marked.size == img.size
        assert marked != img

    def test_different_radius(self):
        img = Image.new("RGB", (500, 500), color=(255, 255, 255))

        marked1 = draw_fixation_marker(img, 250, 250, radius=30)
        marked2 = draw_fixation_marker(img, 250, 250, radius=100)

        assert marked1 != marked2

    def test_custom_colors(self):
        img = Image.new("RGB", (500, 500), color=(255, 255, 255))
        marked = draw_fixation_marker(
            img, 250, 250, radius=50, outline_color=(0, 255, 0), dot_color=(0, 255, 0)
        )

        assert marked.size == img.size
