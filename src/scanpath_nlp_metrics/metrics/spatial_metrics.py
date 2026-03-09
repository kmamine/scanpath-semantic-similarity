"""
Spatial similarity metrics for scanpath comparison.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import euclidean, directed_hausdorff

log = logging.getLogger(__name__)


def scanpath_to_string(
    scanpath: np.ndarray,
    height: int,
    width: int,
    xbins: int = 12,
    ybins: int = 8,
    duration_bins: int = 0,
) -> Tuple[str, List[int]]:
    """Convert scanpath to string representation via spatial binning."""
    if scanpath.shape[1] == 3:
        coords = scanpath[:, :2]
        durations = scanpath[:, 2]
    else:
        coords = scanpath
        durations = np.ones(len(scanpath))

    x_step = width / xbins
    y_step = height / ybins

    bin_ids = []
    for x, y in coords:
        xi = min(int(x / x_step), xbins - 1)
        yi = min(int(y / y_step), ybins - 1)
        bin_id = yi * xbins + xi
        bin_ids.append(bin_id)

    if duration_bins > 0:
        expanded = []
        for bid, dur in zip(bin_ids, durations):
            repeats = max(1, int(dur / duration_bins))
            expanded.extend([bid] * repeats)
        bin_ids = expanded

    chars = [chr(65 + (bid % 26)) + chr(65 + (bid // 26)) for bid in bin_ids]
    string_rep = "".join(chars)

    return string_rep, bin_ids


def dtw_distance(
    scanpath_a: np.ndarray,
    scanpath_b: np.ndarray,
) -> float:
    """Compute Dynamic Time Warping distance between two scanpaths."""
    from fastdtw import fastdtw

    dist, _ = fastdtw(scanpath_a, scanpath_b, dist=euclidean)
    return float(dist)


def hausdorff_distance(
    scanpath_a: np.ndarray,
    scanpath_b: np.ndarray,
) -> float:
    """Compute Hausdorff distance between two scanpaths."""
    a = np.asarray(scanpath_a, dtype=np.float32)
    b = np.asarray(scanpath_b, dtype=np.float32)

    d1 = directed_hausdorff(a, b)[0]
    d2 = directed_hausdorff(b, a)[0]

    return max(d1, d2)


def frechet_distance(
    scanpath_a: np.ndarray,
    scanpath_b: np.ndarray,
) -> float:
    """Compute discrete Frechet distance between two scanpaths."""
    a = np.asarray(scanpath_a, dtype=np.float32)
    b = np.asarray(scanpath_b, dtype=np.float32)

    n, m = len(a), len(b)

    ca = np.full((n, m), -1.0)

    def _c(i: int, j: int) -> float:
        if ca[i, j] > -1:
            return ca[i, j]

        if i == 0 and j == 0:
            ca[i, j] = euclidean(a[0], b[0])
        elif i > 0 and j == 0:
            ca[i, j] = max(_c(i - 1, 0), euclidean(a[i], b[0]))
        elif i == 0 and j > 0:
            ca[i, j] = max(_c(0, j - 1), euclidean(a[0], b[j]))
        else:
            ca[i, j] = max(
                min(_c(i - 1, j), _c(i - 1, j - 1), _c(i, j - 1)), euclidean(a[i], b[j])
            )

        return ca[i, j]

    return _c(n - 1, m - 1)


def levenshtein_distance(
    scanpath_a: np.ndarray,
    scanpath_b: np.ndarray,
    height: int,
    width: int,
    xbins: int = 12,
    ybins: int = 8,
) -> int:
    """Compute Levenshtein (edit) distance between two scanpaths."""
    import editdistance

    str_a, _ = scanpath_to_string(scanpath_a, height, width, xbins, ybins)
    str_b, _ = scanpath_to_string(scanpath_b, height, width, xbins, ybins)

    return editdistance.eval(str_a, str_b)


def tde_distance(
    scanpath_a: np.ndarray,
    scanpath_b: np.ndarray,
    k: int = 3,
    mode: str = "mean",
) -> float:
    """Compute Time-Delay Embedding similarity between scanpaths."""
    a = np.asarray(scanpath_a, dtype=np.float32)
    b = np.asarray(scanpath_b, dtype=np.float32)

    if len(a) < k or len(b) < k:
        log.warning(
            f"TDE requires k < scanpath length. Got k={k}, len(a)={len(a)}, len(b)={len(b)}"
        )
        return float("inf")

    def _create_vectors(sp: np.ndarray, k: int) -> List[np.ndarray]:
        vectors = []
        for i in range(len(sp) - k + 1):
            vectors.append(sp[i : i + k].flatten())
        return vectors

    vecs_a = _create_vectors(a, k)
    vecs_b = _create_vectors(b, k)

    distances = []
    for v_b in vecs_b:
        min_dist = min(np.linalg.norm(v_b - v_a) for v_a in vecs_a)
        distances.append(min_dist / k)

    if mode == "mean":
        return float(np.mean(distances))
    elif mode == "hausdorff":
        return float(np.max(distances))
    else:
        raise ValueError(f"Unknown mode: {mode}")


class ScanMatch:
    """ScanMatch algorithm for scanpath comparison."""

    def __init__(
        self,
        xres: int,
        yres: int,
        xbin: int = 8,
        ybin: int = 6,
        threshold: float = 3.5,
        gap_value: float = 0.0,
        temp_bin: float = 0.0,
        offset: Tuple[int, int] = (0, 0),
    ):
        self.xres = xres
        self.yres = yres
        self.xbin = xbin
        self.ybin = ybin
        self.threshold = threshold
        self.gap_value = gap_value
        self.temp_bin = temp_bin
        self.offset = offset

        self._create_substitution_matrix()
        self._create_grid_mask()

    def _create_substitution_matrix(self) -> None:
        """Create substitution matrix for sequence alignment."""
        n_bins = self.xbin * self.ybin
        self.sub_matrix = np.zeros((n_bins, n_bins))

        for i in range(self.ybin):
            for j in range(self.xbin):
                for ii in range(self.ybin):
                    for jj in range(self.xbin):
                        idx1 = i * self.xbin + j
                        idx2 = ii * self.xbin + jj
                        dist = np.sqrt((j - jj) ** 2 + (i - ii) ** 2)
                        self.sub_matrix[idx1, idx2] = dist

        max_dist = np.max(self.sub_matrix)
        self.sub_matrix = np.abs(self.sub_matrix - max_dist) - (
            max_dist - self.threshold
        )

    def _create_grid_mask(self) -> None:
        """Create grid mask for binning coordinates."""
        x_step = self.xres / self.xbin
        y_step = self.yres / self.ybin

        self.grid_mask = np.zeros((self.yres, self.xres), dtype=np.int32)

        for y in range(self.yres):
            for x in range(self.xres):
                xi = min(int(x / x_step), self.xbin - 1)
                yi = min(int(y / y_step), self.ybin - 1)
                self.grid_mask[y, x] = yi * self.xbin + xi

    def scanpath_to_sequence(
        self,
        scanpath: np.ndarray,
    ) -> np.ndarray:
        """Convert scanpath coordinates to bin sequence."""
        if scanpath.shape[1] == 3:
            coords = scanpath[:, :2]
            durations = scanpath[:, 2]
        else:
            coords = scanpath
            durations = None

        coords = coords - np.array(self.offset)
        coords = np.clip(coords, 0, [self.xres - 1, self.yres - 1])
        coords = coords.astype(np.int32)

        bin_ids = self.grid_mask[coords[:, 1], coords[:, 0]]

        if self.temp_bin > 0 and durations is not None:
            expanded = []
            for bid, dur in zip(bin_ids, durations):
                repeats = max(1, int(dur / self.temp_bin))
                expanded.extend([bid] * repeats)
            bin_ids = np.array(expanded)

        return bin_ids

    def match(
        self,
        seq_a: np.ndarray,
        seq_b: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute ScanMatch score between two bin sequences."""
        n, m = len(seq_a), len(seq_b)

        F = np.zeros((n + 1, m + 1))
        for i in range(n + 1):
            F[i, 0] = self.gap_value * i
        for j in range(m + 1):
            F[0, j] = self.gap_value * j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = F[i - 1, j - 1] + self.sub_matrix[seq_a[i - 1], seq_b[j - 1]]
                delete = F[i - 1, j] + self.gap_value
                insert = F[i, j - 1] + self.gap_value
                F[i, j] = max(match, insert, delete)

        max_score = np.max(F)
        max_sub = np.max(self.sub_matrix)
        scale = max_sub * max(n, m)
        normalized_score = max_score / scale if scale > 0 else 0.0

        return normalized_score, F, seq_a, seq_b

    def score(
        self,
        scanpath_a: np.ndarray,
        scanpath_b: np.ndarray,
    ) -> float:
        """Compute ScanMatch score between two scanpaths."""
        seq_a = self.scanpath_to_sequence(scanpath_a)
        seq_b = self.scanpath_to_sequence(scanpath_b)

        if len(seq_a) == 0 or len(seq_b) == 0:
            return 0.0

        score, _, _, _ = self.match(seq_a, seq_b)
        return score


def compute_pairwise_spatial(
    scanpaths: List[np.ndarray],
    metric_fn,
    **kwargs,
) -> np.ndarray:
    """Compute pairwise spatial metric for a list of scanpaths."""
    n = len(scanpaths)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            score = metric_fn(scanpaths[i], scanpaths[j], **kwargs)
            matrix[i, j] = score
            matrix[j, i] = score

    if metric_fn.__name__ in ["score"]:
        np.fill_diagonal(matrix, 1.0)
    else:
        np.fill_diagonal(matrix, 0.0)

    return matrix


def normalize_distance_matrix(
    matrix: np.ndarray,
    invert: bool = True,
) -> np.ndarray:
    """Normalize a distance matrix to [0, 1] range."""
    max_val = np.max(matrix[np.triu_indices(len(matrix), k=1)])
    if max_val > 0:
        normalized = matrix / max_val
    else:
        normalized = matrix.copy()

    if invert:
        normalized = 1.0 - normalized

    np.fill_diagonal(normalized, 1.0 if invert else 0.0)
    return normalized
