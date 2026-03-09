"""
Pairwise metric computation utilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data.loader import Scanpath
from ..config import Config
from ..metrics.nlp_metrics import (
    compute_pairwise_rouge,
    compute_pairwise_bleu,
    compute_pairwise_bertscore,
    compute_pairwise_bm25,
)
from ..metrics.spatial_metrics import (
    ScanMatch,
    dtw_distance,
    hausdorff_distance,
    tde_distance,
    levenshtein_distance,
    normalize_distance_matrix,
)
from ..metrics.multimatch import compute_pairwise_multimatch

log = logging.getLogger(__name__)


@dataclass
class PairwiseResult:
    """Container for pairwise comparison results."""

    image_id: str
    pair_i: int
    pair_j: int

    rouge: float
    bleu: float
    bertscore: float
    bm25: float

    scanmatch: float
    dtw: float
    tde: float
    hausdorff: float
    levenshtein: float

    mm_vector: float
    mm_direction: float
    mm_length: float
    mm_position: float
    mm_duration: float
    mm_average: float

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "pair_i": self.pair_i,
            "pair_j": self.pair_j,
            "rouge": self.rouge,
            "bleu": self.bleu,
            "bertscore": self.bertscore,
            "bm25": self.bm25,
            "scanmatch": self.scanmatch,
            "dtw": self.dtw,
            "tde": self.tde,
            "hausdorff": self.hausdorff,
            "levenshtein": self.levenshtein,
            "mm_vector": self.mm_vector,
            "mm_direction": self.mm_direction,
            "mm_length": self.mm_length,
            "mm_position": self.mm_position,
            "mm_duration": self.mm_duration,
            "mm_average": self.mm_average,
        }


class PairwiseComparator:
    """Computes all pairwise metrics for scanpaths within images."""

    def __init__(
        self,
        config: Config,
        compute_multimatch: bool = True,
    ):
        self.config = config
        self.compute_multimatch = compute_multimatch

        self.scanmatch = ScanMatch(
            xres=config.scanmatch.xres,
            yres=config.scanmatch.yres,
            xbin=config.scanmatch.xbin,
            ybin=config.scanmatch.ybin,
            threshold=config.scanmatch.threshold,
            gap_value=config.scanmatch.gap_value,
            temp_bin=config.scanmatch.temp_bin,
            offset=config.scanmatch.offset,
        )

    def compute_for_image(
        self,
        image_id: str,
        scanpaths: List[Scanpath],
        descriptions: List[str],
    ) -> List[PairwiseResult]:
        """Compute all pairwise metrics for scanpaths within a single image."""
        n = len(scanpaths)

        if n != len(descriptions):
            raise ValueError(
                f"Mismatch: {n} scanpaths but {len(descriptions)} descriptions"
            )

        coords = [sp.to_array() for sp in scanpaths]

        log.info(f"Computing NLP metrics for {image_id}...")
        rouge_matrix = compute_pairwise_rouge(descriptions)
        bleu_matrix = compute_pairwise_bleu(descriptions)
        bertscore_matrix = compute_pairwise_bertscore(descriptions)
        bm25_matrix = compute_pairwise_bm25(descriptions)

        log.info(f"Computing spatial metrics for {image_id}...")
        scanmatch_matrix = self._compute_scanmatch(coords)
        dtw_matrix = self._compute_dtw(coords)
        tde_matrix = self._compute_tde(coords)
        hausdorff_matrix = self._compute_hausdorff(coords)
        levenshtein_matrix = self._compute_levenshtein(coords)

        if self.compute_multimatch:
            log.info(f"Computing MultiMatch for {image_id}...")
            mm_matrices = compute_pairwise_multimatch(
                coords,
                screensize=(
                    self.config.dataset.image_width,
                    self.config.dataset.image_height,
                ),
            )
            mm_vector = mm_matrices[:, :, 0]
            mm_direction = mm_matrices[:, :, 1]
            mm_length = mm_matrices[:, :, 2]
            mm_position = mm_matrices[:, :, 3]
            mm_duration = mm_matrices[:, :, 4]
            mm_average = np.nanmean(mm_matrices, axis=2)
        else:
            mm_vector = np.full((n, n), np.nan)
            mm_direction = np.full((n, n), np.nan)
            mm_length = np.full((n, n), np.nan)
            mm_position = np.full((n, n), np.nan)
            mm_duration = np.full((n, n), np.nan)
            mm_average = np.full((n, n), np.nan)

        results = []
        for i, j in combinations(range(n), 2):
            result = PairwiseResult(
                image_id=image_id,
                pair_i=i,
                pair_j=j,
                rouge=rouge_matrix[i, j],
                bleu=bleu_matrix[i, j],
                bertscore=bertscore_matrix[i, j],
                bm25=bm25_matrix[i, j],
                scanmatch=scanmatch_matrix[i, j],
                dtw=dtw_matrix[i, j],
                tde=tde_matrix[i, j],
                hausdorff=hausdorff_matrix[i, j],
                levenshtein=levenshtein_matrix[i, j],
                mm_vector=mm_vector[i, j],
                mm_direction=mm_direction[i, j],
                mm_length=mm_length[i, j],
                mm_position=mm_position[i, j],
                mm_duration=mm_duration[i, j],
                mm_average=mm_average[i, j],
            )
            results.append(result)

        return results

    def _compute_scanmatch(self, coords: List[np.ndarray]) -> np.ndarray:
        n = len(coords)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                score = self.scanmatch.score(coords[i], coords[j])
                matrix[i, j] = score
                matrix[j, i] = score
        np.fill_diagonal(matrix, 1.0)
        return matrix

    def _compute_dtw(self, coords: List[np.ndarray]) -> np.ndarray:
        n = len(coords)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                dist = dtw_distance(coords[i][:, :2], coords[j][:, :2])
                matrix[i, j] = dist
                matrix[j, i] = dist
        return normalize_distance_matrix(matrix, invert=True)

    def _compute_tde(self, coords: List[np.ndarray]) -> np.ndarray:
        n = len(coords)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    dist = tde_distance(coords[i][:, :2], coords[j][:, :2], k=3)
                    matrix[i, j] = dist
                    matrix[j, i] = dist
                except Exception:
                    matrix[i, j] = np.nan
                    matrix[j, i] = np.nan
        return normalize_distance_matrix(matrix, invert=True)

    def _compute_hausdorff(self, coords: List[np.ndarray]) -> np.ndarray:
        n = len(coords)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                dist = hausdorff_distance(coords[i][:, :2], coords[j][:, :2])
                matrix[i, j] = dist
                matrix[j, i] = dist
        return normalize_distance_matrix(matrix, invert=True)

    def _compute_levenshtein(self, coords: List[np.ndarray]) -> np.ndarray:
        n = len(coords)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                dist = levenshtein_distance(
                    coords[i],
                    coords[j],
                    height=self.config.dataset.image_height,
                    width=self.config.dataset.image_width,
                )
                matrix[i, j] = dist
                matrix[j, i] = dist
        return normalize_distance_matrix(matrix, invert=True)

    def compute_all(
        self,
        scanpaths: Dict[str, List[Scanpath]],
        descriptions: Dict[str, List[str]],
    ) -> pd.DataFrame:
        """Compute pairwise metrics for all images."""
        all_results = []

        image_ids = list(scanpaths.keys())
        for image_id in tqdm(image_ids, desc="Computing metrics"):
            sp_list = scanpaths[image_id]
            desc_list = descriptions.get(image_id, [])

            if len(desc_list) != len(sp_list):
                log.warning(f"Skipping {image_id}: description count mismatch")
                continue

            results = self.compute_for_image(image_id, sp_list, desc_list)
            all_results.extend(results)

        df = pd.DataFrame([r.to_dict() for r in all_results])
        return df


def load_descriptions_from_jsonl(path: Path) -> Dict[str, List[str]]:
    """Load scanpath descriptions from JSONL file."""
    descriptions: Dict[str, List[str]] = {}

    with open(path) as f:
        for line in f:
            record = json.loads(line)
            image_id = record["image_id"]
            scanpath_idx = record["scanpath_idx"]
            desc = record["description"]

            if image_id not in descriptions:
                descriptions[image_id] = []

            while len(descriptions[image_id]) <= scanpath_idx:
                descriptions[image_id].append("")

            descriptions[image_id][scanpath_idx] = desc

    return descriptions
