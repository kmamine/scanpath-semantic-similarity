"""
Patch-based scanpath description generation.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Union

import numpy as np
from PIL import Image

from .vlm_client import VLMClient
from ..data.patch_extraction import extract_patch

log = logging.getLogger(__name__)

ScanpathType = List[List[float]]  # [[x, y, duration], ...]


class PatchGenerator:
    """Generate descriptions using patch-based method."""

    def __init__(
        self,
        vlm_client: VLMClient,
        patch_size: int = 96,
        max_tokens_fixation: int = 80,
        max_tokens_scanpath: int = 180,
        temperature: float = 0.2,
        fixation_prompt: str = None,
        scanpath_prompt_template: str = None,
    ):
        self.vlm = vlm_client
        self.patch_size = patch_size
        self.max_tokens_fixation = max_tokens_fixation
        self.max_tokens_scanpath = max_tokens_scanpath
        self.temperature = temperature

        self.fixation_prompt = fixation_prompt or (
            "Describe what you see in this image patch in 1-2 sentences. "
            "Focus on any objects, faces, text, or salient visual content. "
            "If the patch appears blurry or shows only texture/background, "
            "describe the dominant colour, texture, or any partial object visible."
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
        Generate description for a scanpath.

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

            patch = extract_patch(image, x, y, self.patch_size)

            response = self.vlm.describe(
                patch,
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
