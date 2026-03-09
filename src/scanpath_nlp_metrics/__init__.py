"""
scanpath_nlp_metrics - NLP and spatial metrics for scanpath comparison.

Main class: ScanpathComparator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .generation.vlm_client import VLMClient
from .generation.patch import PatchGenerator
from .generation.marker import MarkerGenerator
from .metrics.nlp import compute_nlp_metrics
from .metrics.spatial_metrics import (
    ScanMatch,
    dtw_distance,
    hausdorff_distance,
    levenshtein_distance,
    tde_distance,
    normalize_distance,
)
from .metrics.multimatch import compute_multimatch

log = logging.getLogger(__name__)

ScanpathType = List[List[float]]  # [[x, y, duration], ...]

DEFAULT_METRICS = [
    "rouge",
    "bleu",
    "bert_score",
    "bm25",
    "scanmatch",
    "dtw",
    "hausdorff",
    "levenshtein",
    "tde",
    "multimatch_vector",
    "multimatch_direction",
    "multimatch_length",
    "multimatch_position",
    "multimatch_duration",
    "multimatch_mean",
]


@dataclass
class ScanpathComparator:
    """
    Compare two scanpaths using VLM-generated descriptions and spatial metrics.

    Usage:
        comparator = ScanpathComparator(
            vlm_base_url="http://localhost:8000/v1",
            vlm_api_key="your-key",
            vlm_model="qwen2-vl-7b-instruct",
            method="patch",
            patch_size=96,
            screen_size=(1680, 1050),
        )

        result = comparator.compare(
            scanpath_a=[[x1, y1, d1], [x2, y2, d2], ...],
            scanpath_b=[[x1, y1, d1], [x2, y2, d2], ...],
            image="path/to/image.jpg",
        )
    """

    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_api_key: str = "x"
    vlm_model: str = "qwen2-vl-7b-instruct"
    vlm_retries: int = 3

    method: str = "patch"  # "patch" or "marker"
    patch_size: int = 96
    marker_radius: int = 100

    screen_size: Tuple[int, int] = (1680, 1050)

    max_tokens_fixation: int = 80
    max_tokens_scanpath: int = 180
    temperature: float = 0.2

    metrics: Optional[List[str]] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = DEFAULT_METRICS.copy()

        self.vlm_client = VLMClient(
            base_url=self.vlm_base_url,
            api_key=self.vlm_api_key,
            model=self.vlm_model,
            retries=self.vlm_retries,
        )

        if self.method == "patch":
            self.generator = PatchGenerator(
                vlm_client=self.vlm_client,
                patch_size=self.patch_size,
                max_tokens_fixation=self.max_tokens_fixation,
                max_tokens_scanpath=self.max_tokens_scanpath,
                temperature=self.temperature,
            )
        elif self.method == "marker":
            self.generator = MarkerGenerator(
                vlm_client=self.vlm_client,
                marker_radius=self.marker_radius,
                max_tokens_fixation=self.max_tokens_fixation,
                max_tokens_scanpath=self.max_tokens_scanpath,
                temperature=self.temperature,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'patch' or 'marker'")

        self.scanmatch = ScanMatch(
            xres=self.screen_size[0],
            yres=self.screen_size[1],
            xbin=14,
            ybin=9,
        )

    def _compute_spatial_metrics(
        self,
        scanpath_a: ScanpathType,
        scanpath_b: ScanpathType,
    ) -> Dict[str, float]:
        """Compute spatial metrics between two scanpaths."""
        results = {}

        coords_a = np.array(scanpath_a, dtype=np.float32)
        coords_b = np.array(scanpath_b, dtype=np.float32)

        if "scanmatch" in self.metrics:
            results["scanmatch"] = self.scanmatch.score(coords_a, coords_b)

        if "dtw" in self.metrics:
            dist = dtw_distance(coords_a[:, :2], coords_b[:, :2])
            results["dtw"] = normalize_distance(dist)

        if "hausdorff" in self.metrics:
            dist = hausdorff_distance(coords_a[:, :2], coords_b[:, :2])
            results["hausdorff"] = normalize_distance(dist)

        if "levenshtein" in self.metrics:
            dist = levenshtein_distance(
                coords_a,
                coords_b,
                height=self.screen_size[1],
                width=self.screen_size[0],
            )
            results["levenshtein"] = normalize_distance(dist)

        if "tde" in self.metrics:
            dist = tde_distance(coords_a[:, :2], coords_b[:, :2], k=3)
            results["tde"] = normalize_distance(dist)

        return results

    def _compute_multimatch(
        self,
        scanpath_a: ScanpathType,
        scanpath_b: ScanpathType,
    ) -> Dict[str, float]:
        """Compute MultiMatch metrics."""
        if not any(m.startswith("multimatch") for m in self.metrics):
            return {}

        coords_a = np.array(scanpath_a, dtype=np.float32)
        coords_b = np.array(scanpath_b, dtype=np.float32)

        fix_arr_a = np.array(
            [(f[0], f[1], f[2] if len(f) > 2 else 0.0) for f in coords_a],
            dtype=[("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")],
        )
        fix_arr_b = np.array(
            [(f[0], f[1], f[2] if len(f) > 2 else 0.0) for f in coords_b],
            dtype=[("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")],
        )

        result = compute_multimatch(
            fix_arr_a,
            fix_arr_b,
            screensize=list(self.screen_size),
        )

        results = {}
        if "multimatch_vector" in self.metrics:
            results["multimatch_vector"] = result.vector
        if "multimatch_direction" in self.metrics:
            results["multimatch_direction"] = result.direction
        if "multimatch_length" in self.metrics:
            results["multimatch_length"] = result.length
        if "multimatch_position" in self.metrics:
            results["multimatch_position"] = result.position
        if "multimatch_duration" in self.metrics:
            results["multimatch_duration"] = result.duration
        if "multimatch_mean" in self.metrics:
            results["multimatch_mean"] = (
                result.vector
                + result.direction
                + result.length
                + result.position
                + result.duration
            ) / 5

        return results

    def _compute_nlp_metrics(
        self,
        description_a: str,
        description_b: str,
    ) -> Dict[str, float]:
        """Compute NLP metrics between two descriptions."""
        nlp_metric_names = ["rouge", "bleu", "bert_score", "bm25"]
        requested = [m for m in nlp_metric_names if m in self.metrics]

        if not requested:
            return {}

        return compute_nlp_metrics(description_a, description_b, requested)

    def compare(
        self,
        scanpath_a: ScanpathType,
        scanpath_b: ScanpathType,
        image: Union[str, Image.Image],
    ) -> Dict[str, any]:
        """
        Compare two scanpaths on the same image.

        Args:
            scanpath_a: First scanpath as [[x, y, duration], ...]
            scanpath_b: Second scanpath as [[x, y, duration], ...]
            image: Path to image or PIL Image

        Returns:
            Dict containing:
            - description_a: VLM description for scanpath A
            - description_b: VLM description for scanpath B
            - All requested metrics
        """
        description_a, _ = self.generator.generate_description(image, scanpath_a)
        description_b, _ = self.generator.generate_description(image, scanpath_b)

        results = {
            "description_a": description_a,
            "description_b": description_b,
        }

        nlp_results = self._compute_nlp_metrics(description_a, description_b)
        results.update(nlp_results)

        spatial_results = self._compute_spatial_metrics(scanpath_a, scanpath_b)
        results.update(spatial_results)

        multimatch_results = self._compute_multimatch(scanpath_a, scanpath_b)
        results.update(multimatch_results)

        return results


__version__ = "0.0.1"

__all__ = [
    "ScanpathComparator",
    "ScanpathType",
    "DEFAULT_METRICS",
]
