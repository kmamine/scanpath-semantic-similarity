"""
Generation module for scanpath descriptions using VLMs.
"""

from .vlm_client import VLMClient, VLMResponse
from .patch import PatchGenerator
from .marker import MarkerGenerator

__all__ = [
    "VLMClient",
    "VLMResponse",
    "PatchGenerator",
    "MarkerGenerator",
]
