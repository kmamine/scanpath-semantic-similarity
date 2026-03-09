"""
Generation module for scanpath descriptions using VLMs.
"""

from .vlm_client import VLMClient, VLMResponse
from .description_pipeline import DescriptionPipeline
from .marker_pipeline import MarkerPipeline

__all__ = [
    "VLMClient",
    "VLMResponse",
    "DescriptionPipeline",
    "MarkerPipeline",
]
