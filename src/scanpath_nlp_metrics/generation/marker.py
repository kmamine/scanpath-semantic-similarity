"""
Marker-based scanpath description generation.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Union

from PIL import Image

from .vlm_client import VLMClient
from ..data.image_marking import draw_fixation_marker

log = logging.getLogger(__name__)

ScanpathType = List[List[float]]  # [[x, y, duration], ...]


class MarkerGenerator:
    """Generate descriptions using full-image with fixation markers."""

    def __init__(
        self,
        vlm_client: VLMClient,
        marker_radius: int = 100,
        outline_width: int = 3,
        dot_radius: int = 5,
        max_tokens_fixation: int = 80,
        max_tokens_scanpath: int = 180,
        temperature: float = 0.2,
        fixation_prompt: str = None,
        scanpath_prompt_template: str = None,
    ):
        self.vlm = vlm_client
        self.marker_radius = marker_radius
        self.outline_width = outline_width
        self.dot_radius = dot_radius
        self.max_tokens_fixation = max_tokens_fixation
        self.max_tokens_scanpath = max_tokens_scanpath
        self.temperature = temperature

        self.fixation_prompt = fixation_prompt or (
            "You are analyzing where a viewer looked at an image. "
            "The red circle marks the region they fixated on (the circle center is the exact gaze point).\n\n"
            "Describe what is inside the circled region in 1-2 sentences. Focus on:\n"
            "- Objects or elements within the circle\n"
            "- The visual content at the fixation location\n"
            "- How this region relates to the broader image context\n\n"
            "Be specific about what the viewer was looking at in that circled area."
        )

        self.scanpath_prompt_template = scanpath_prompt_template or (
            "You are analysing where a human viewer looked at an image. "
            "Below are sequential descriptions of the image regions they fixated on (in temporal order):\n\n"
            "{fixation_list}\n\n"
            "Given the full image provided and these fixation descriptions, write a single coherent paragraph "
            "summarising what this viewer attended to and what cognitive strategy they might have used."
        )

    def generate_description(
        self,
        image: Union[str, Image.Image],
        scanpath: ScanpathType,
    ) -> Tuple[str, List[str]]:
        """
        Generate description for a scanpath using marker method.

        Args:
            image: Path to image or PIL Image
            scanpath: List of [x, y, duration] fixations

        Returns:
            Tuple of (scanpath_description, list_of_fixation_descriptions)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        fixation_descriptions = []

        for fixation in scanpath:
            x, y = fixation[0], fixation[1]

            marked_image = draw_fixation_marker(
                image,
                x,
                y,
                radius=self.marker_radius,
                outline_width=self.outline_width,
                dot_radius=self.dot_radius,
            )

            response = self.vlm.describe(
                marked_image,
                self.fixation_prompt,
                max_tokens=self.max_tokens_fixation,
                temperature=self.temperature,
            )

            fixation_descriptions.append(response.content)

        fix_list = "\n".join(
            f"Fixation {i + 1}: {desc}" for i, desc in enumerate(fixation_descriptions)
        )

        scanpath_description = self.vlm.describe(
            image,
            self.scanpath_prompt_template.format(fixation_list=fix_list),
            max_tokens=self.max_tokens_scanpath,
            temperature=self.temperature,
        )

        return scanpath_description.content, fixation_descriptions
