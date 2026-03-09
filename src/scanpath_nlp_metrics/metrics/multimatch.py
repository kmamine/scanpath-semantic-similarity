"""
MultiMatch implementation for scanpath comparison.

Based on the MultiMatch algorithm by Jarodzka et al., 2010 and Dewhurst et al., 2012.
Self-contained implementation without external multimatch-gaze dependency.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)


def cart2pol(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transform cartesian into polar coordinates."""
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return rho, theta


def calcangle(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate angle between two vectors (saccades)."""
    angle = math.degrees(
        math.acos(np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2)))
    )
    return angle


def _get_empty_path() -> dict:
    return dict(
        fix=dict(dur=[]),
        sac=dict(
            x=[],
            y=[],
            lenx=[],
            leny=[],
            theta=[],
            rho=[],
        ),
    )


def keepsaccade(i: int, j: int, sim: dict, data: dict) -> Tuple[int, int]:
    """Helper function for scanpath simplification."""
    for t, k in (
        ("sac", "lenx"),
        ("sac", "leny"),
        ("sac", "x"),
        ("sac", "y"),
        ("sac", "theta"),
        ("sac", "rho"),
        ("fix", "dur"),
    ):
        sim[t][k].insert(j, data[t][k][i])
    return i + 1, j + 1


def gen_scanpath_structure(data: np.ndarray) -> dict:
    """Transform fixation vector into vector-based scanpath representation."""
    fixations = dict(x=data["start_x"], y=data["start_y"], dur=data["duration"])
    lenx = np.diff(data["start_x"])
    leny = np.diff(data["start_y"])
    rho, theta = cart2pol(lenx, leny)

    saccades = dict(
        x=data[:-1]["start_x"],
        y=data[:-1]["start_y"],
        lenx=lenx,
        leny=leny,
        theta=theta,
        rho=rho,
    )
    return dict(fix=fixations, sac=saccades)


def simlen(path: dict, TAmp: float, TDur: float) -> dict:
    """Simplify scanpaths based on saccadic length."""
    saccades = path["sac"]
    fixations = path["fix"]

    if len(saccades["x"]) < 1:
        return path

    i = 0
    j = 0
    sim = _get_empty_path()

    while i <= len(saccades["x"]) - 1:
        if i == len(saccades["x"]) - 1:
            if saccades["rho"][i] < TAmp:
                if (fixations["dur"][-1] < TDur) or (fixations["dur"][-2] < TDur):
                    v_x = saccades["lenx"][-2] + saccades["lenx"][-1]
                    v_y = saccades["leny"][-2] + saccades["leny"][-1]
                    rho, theta = cart2pol(v_x, v_y)
                    sim["sac"]["lenx"][j - 1] = v_x
                    sim["sac"]["leny"][j - 1] = v_y
                    sim["sac"]["theta"][j - 1] = theta
                    sim["sac"]["rho"][j - 1] = rho
                    sim["fix"]["dur"].insert(j, fixations["dur"][i - 1])
                    j -= 1
                    i += 1
                else:
                    i, j = keepsaccade(i, j, sim, path)
            else:
                i, j = keepsaccade(i, j, sim, path)
        else:
            if (saccades["rho"][i] < TAmp) and (i < len(saccades["x"]) - 1):
                if (fixations["dur"][i + 1] < TDur) or (fixations["dur"][i] < TDur):
                    v_x = saccades["lenx"][i] + saccades["lenx"][i + 1]
                    v_y = saccades["leny"][i] + saccades["leny"][i + 1]
                    rho, theta = cart2pol(v_x, v_y)
                    sim["sac"]["lenx"].insert(j, v_x)
                    sim["sac"]["leny"].insert(j, v_y)
                    sim["sac"]["x"].insert(j, saccades["x"][i])
                    sim["sac"]["y"].insert(j, saccades["y"][i])
                    sim["sac"]["theta"].insert(j, theta)
                    sim["sac"]["rho"].insert(j, rho)
                    sim["fix"]["dur"].insert(j, fixations["dur"][i])
                    i += 2
                    j += 1
                else:
                    i, j = keepsaccade(i, j, sim, path)
            else:
                i, j = keepsaccade(i, j, sim, path)

    sim["fix"]["dur"].append(fixations["dur"][-1])
    return sim


def simdir(path: dict, TDir: float, TDur: float) -> dict:
    """Simplify scanpaths based on angular relations between saccades."""
    saccades = path["sac"]
    fixations = path["fix"]

    if len(saccades["x"]) < 1:
        return path

    i = 0
    j = 0
    sim = _get_empty_path()

    while i <= len(saccades["x"]) - 1:
        if i < len(saccades["x"]) - 1:
            v1 = [saccades["lenx"][i], saccades["leny"][i]]
            v2 = [saccades["lenx"][i + 1], saccades["leny"][i + 1]]
            angle = calcangle(v1, v2)
        else:
            angle = float("inf")

        if (angle < TDir) & (i < len(saccades["x"]) - 1):
            if fixations["dur"][i + 1] < TDur:
                v_x = saccades["lenx"][i] + saccades["lenx"][i + 1]
                v_y = saccades["leny"][i] + saccades["leny"][i + 1]
                rho, theta = cart2pol(v_x, v_y)
                sim["sac"]["lenx"].insert(j, v_x)
                sim["sac"]["leny"].insert(j, v_y)
                sim["sac"]["x"].insert(j, saccades["x"][i])
                sim["sac"]["y"].insert(j, saccades["y"][i])
                sim["sac"]["theta"].insert(j, theta)
                sim["sac"]["rho"].insert(j, rho)
                sim["fix"]["dur"].insert(j, fixations["dur"][i])
                i += 2
                j += 1
            else:
                i, j = keepsaccade(i, j, sim, path)
        else:
            i, j = keepsaccade(i, j, sim, path)

    sim["fix"]["dur"].append(fixations["dur"][-1])
    return sim


def simplify_scanpath(path: dict, TAmp: float, TDir: float, TDur: float) -> dict:
    """Simplify scanpaths until no further simplification is possible."""
    prev_length = len(path["fix"]["dur"])
    while True:
        path = simdir(path, TDir, TDur)
        path = simlen(path, TAmp, TDur)
        length = len(path["fix"]["dur"])
        if length == prev_length:
            return path
        else:
            prev_length = length


def cal_vectordifferences(path1: dict, path2: dict) -> np.ndarray:
    """Create matrix of vector-length differences of all vector pairs."""
    x1 = np.asarray(path1["sac"]["lenx"])
    x2 = np.asarray(path2["sac"]["lenx"])
    y1 = np.asarray(path1["sac"]["leny"])
    y2 = np.asarray(path2["sac"]["leny"])

    rows = []
    for i in range(0, len(x1)):
        x_diff = abs(x1[i] * np.ones(len(x2)) - x2)
        y_diff = abs(y1[i] * np.ones(len(y2)) - y2)
        rows.append(np.asarray(np.sqrt(x_diff**2 + y_diff**2)))
    M = np.vstack(rows)
    return M


def createdirectedgraph(
    scanpath_dim: Tuple[int, int],
    M: np.ndarray,
    M_assignment: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Create a directed graph for Dijkstra algorithm."""
    rows = []
    cols = []
    weight = []

    for i in range(0, scanpath_dim[0]):
        for j in range(0, scanpath_dim[1]):
            currentNode = i * scanpath_dim[1] + j
            if (i == scanpath_dim[0] - 1) & (j < scanpath_dim[1] - 1):
                rows.append(currentNode)
                cols.append(currentNode + 1)
                weight.append(M[i, j + 1])
            elif (i < scanpath_dim[0] - 1) & (j == scanpath_dim[1] - 1):
                rows.append(currentNode)
                cols.append(currentNode + scanpath_dim[1])
                weight.append(M[i + 1, j])
            elif (i == scanpath_dim[0] - 1) & (j == scanpath_dim[1] - 1):
                rows.append(currentNode)
                cols.append(currentNode)
                weight.append(0)
            else:
                rows.append(currentNode)
                rows.append(currentNode)
                rows.append(currentNode)
                cols.append(currentNode + 1)
                cols.append(currentNode + scanpath_dim[1])
                cols.append(currentNode + scanpath_dim[1] + 1)
                weight.append(M[i, j + 1])
                weight.append(M[i + 1, j])
                weight.append(M[i + 1, j + 1])

    rows = np.asarray(rows)
    cols = np.asarray(cols)
    weight = np.asarray(weight)
    numVert = scanpath_dim[0] * scanpath_dim[1]
    return numVert, rows, cols, weight


def dijkstra(
    numVert: int,
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    start: int,
    end: int,
) -> Tuple[List[int], float]:
    """Dijkstra algorithm for shortest path."""
    arrayWeightedGraph = (
        sp.coo_matrix((data, (rows, cols)), shape=(numVert, numVert))
    ).tocsr()

    dist_matrix, predecessors = sp.csgraph.dijkstra(
        csgraph=arrayWeightedGraph, directed=True, indices=0, return_predecessors=True
    )

    path = [end]
    dist = float(dist_matrix[end])
    while end != -9999:
        path.append(predecessors[end])
        end = predecessors[end]

    return path[-2::-1], dist


def cal_angulardifference(
    data1: dict,
    data2: dict,
    path: List[int],
    M_assignment: np.ndarray,
) -> List[float]:
    """Calculate angular similarity of two scanpaths."""
    theta1 = data1["sac"]["theta"]
    theta2 = data2["sac"]["theta"]
    anglediff = []

    for p in path:
        i, j = np.where(M_assignment == p)
        spT = [theta1[i.item()], theta2[j.item()]]
        for t in range(0, len(spT)):
            if spT[t] < 0:
                spT[t] = math.pi + (math.pi + spT[t])
        spT = abs(spT[0] - spT[1])
        if spT > math.pi:
            spT = 2 * math.pi - spT
        anglediff.append(spT)
    return anglediff


def cal_durationdifference(
    data1: dict,
    data2: dict,
    path: List[int],
    M_assignment: np.ndarray,
) -> List[float]:
    """Calculate similarity of fixation durations."""
    dur1 = data1["fix"]["dur"]
    dur2 = data2["fix"]["dur"]
    durdiff = []

    for p in path:
        i, j = np.where(M_assignment == p)
        maxlist = [dur1[i.item()], dur2[j.item()]]
        durdiff.append(abs(dur1[i.item()] - dur2[j.item()]) / abs(max(maxlist)))
    return durdiff


def cal_lengthdifference(
    data1: dict,
    data2: dict,
    path: List[int],
    M_assignment: np.ndarray,
) -> List[float]:
    """Calculate length similarity of two scanpaths."""
    len1 = np.asarray(data1["sac"]["rho"])
    len2 = np.asarray(data2["sac"]["rho"])
    lendiff = []

    for p in path:
        i, j = np.where(M_assignment == p)
        lendiff.append(abs(len1[i] - len2[j]))
    return lendiff


def cal_positiondifference(
    data1: dict,
    data2: dict,
    path: List[int],
    M_assignment: np.ndarray,
) -> List[float]:
    """Calculate position similarity of two scanpaths."""
    x1 = np.asarray(data1["sac"]["x"])
    x2 = np.asarray(data2["sac"]["x"])
    y1 = np.asarray(data1["sac"]["y"])
    y2 = np.asarray(data2["sac"]["y"])
    posdiff = []

    for p in path:
        i, j = np.where(M_assignment == p)
        posdiff.append(
            math.sqrt(
                (x1[i.item()] - x2[j.item()]) ** 2 + (y1[i.item()] - y2[j.item()]) ** 2
            )
        )
    return posdiff


def cal_vectordifferencealongpath(
    data1: dict,
    data2: dict,
    path: List[int],
    M_assignment: np.ndarray,
) -> List[float]:
    """Calculate vector similarity of two scanpaths."""
    x1 = np.asarray(data1["sac"]["lenx"])
    x2 = np.asarray(data2["sac"]["lenx"])
    y1 = np.asarray(data1["sac"]["leny"])
    y2 = np.asarray(data2["sac"]["leny"])
    vectordiff = []

    for p in path:
        i, j = np.where(M_assignment == p)
        vectordiff.append(
            np.sqrt(
                (x1[i.item()] - x2[j.item()]) ** 2 + (y1[i.item()] - y2[j.item()]) ** 2
            )
        )
    return vectordiff


def getunnormalised(
    data1: dict,
    data2: dict,
    path: List[int],
    M_assignment: np.ndarray,
) -> List[float]:
    """Calculate unnormalised similarity measures."""
    return [
        np.median(fx(data1, data2, path, M_assignment))
        for fx in (
            cal_vectordifferencealongpath,
            cal_angulardifference,
            cal_lengthdifference,
            cal_positiondifference,
            cal_durationdifference,
        )
    ]


def normaliseresults(unnormalised: List[float], screensize: List[int]) -> List[float]:
    """Normalize similarity measures."""
    VectorSimilarity = 1 - unnormalised[0] / (
        2 * math.sqrt(screensize[0] ** 2 + screensize[1] ** 2)
    )
    DirectionSimilarity = 1 - unnormalised[1] / math.pi
    LengthSimilarity = 1 - unnormalised[2] / math.sqrt(
        screensize[0] ** 2 + screensize[1] ** 2
    )
    PositionSimilarity = 1 - unnormalised[3] / math.sqrt(
        screensize[0] ** 2 + screensize[1] ** 2
    )
    DurationSimilarity = 1 - unnormalised[4]
    return [
        VectorSimilarity,
        DirectionSimilarity,
        LengthSimilarity,
        PositionSimilarity,
        DurationSimilarity,
    ]


@dataclass
class MultiMatchResult:
    """Result container for MultiMatch computation."""

    vector: float
    direction: float
    length: float
    position: float
    duration: float

    def to_dict(self) -> dict:
        return {
            "vector": self.vector,
            "direction": self.direction,
            "length": self.length,
            "position": self.position,
            "duration": self.duration,
        }


def compute_multimatch(
    fixation_vectors1: np.ndarray,
    fixation_vectors2: np.ndarray,
    screensize: List[int],
    grouping: bool = False,
    TDir: float = 0.0,
    TDur: float = 0.0,
    TAmp: float = 0.0,
) -> MultiMatchResult:
    """
    Compare two scanpaths using MultiMatch algorithm.

    Args:
        fixation_vectors1: Array of shape (N, 3) with x, y, duration
        fixation_vectors2: Array of shape (M, 3) with x, y, duration
        screensize: [width, height] in pixels
        grouping: Enable scanpath simplification
        TDir: Direction threshold for grouping (degrees)
        TDur: Duration threshold for grouping (seconds)
        TAmp: Amplitude threshold for grouping (pixels)

    Returns:
        MultiMatchResult with 5 similarity scores (0-1, higher = more similar)
    """
    if (len(fixation_vectors1) >= 3) & (len(fixation_vectors2) >= 3):
        path1 = gen_scanpath_structure(fixation_vectors1)
        path2 = gen_scanpath_structure(fixation_vectors2)

        if grouping:
            path1 = simplify_scanpath(path1, TAmp, TDir, TDur)
            path2 = simplify_scanpath(path2, TAmp, TDir, TDur)

        M = cal_vectordifferences(path1, path2)
        scanpath_dim = np.shape(M)
        M_assignment = np.arange(scanpath_dim[0] * scanpath_dim[1]).reshape(
            scanpath_dim[0], scanpath_dim[1]
        )

        numVert, rows, cols, weight = createdirectedgraph(scanpath_dim, M, M_assignment)
        path, dist = dijkstra(
            numVert, rows, cols, weight, 0, scanpath_dim[0] * scanpath_dim[1] - 1
        )

        unnormalised = getunnormalised(path1, path2, path, M_assignment)
        normal = normaliseresults(unnormalised, screensize)

        return MultiMatchResult(
            vector=normal[0],
            direction=normal[1],
            length=normal[2],
            position=normal[3],
            duration=normal[4],
        )
    else:
        return MultiMatchResult(
            vector=float("nan"),
            direction=float("nan"),
            length=float("nan"),
            position=float("nan"),
            duration=float("nan"),
        )


def compute_pairwise_multimatch(
    scanpaths: List[np.ndarray],
    screensize: List[int] = [1680, 1050],
    grouping: bool = False,
    TDir: float = 0.0,
    TDur: float = 0.0,
    TAmp: float = 0.0,
) -> np.ndarray:
    """
    Compute pairwise MultiMatch scores.

    Returns:
        Array of shape (n, n, 5) with similarity scores
    """
    n = len(scanpaths)
    results = np.full((n, n, 5), float("nan"), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            fix1 = scanpaths[i]
            fix2 = scanpaths[j]

            fix_arr1 = np.array(
                [(f.x, f.y, f.duration) for f in fix1],
                dtype=[("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")],
            )
            fix_arr2 = np.array(
                [(f.x, f.y, f.duration) for f in fix2],
                dtype=[("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")],
            )

            result = compute_multimatch(
                fix_arr1, fix_arr2, screensize, grouping, TDir, TDur, TAmp
            )
            results[i, j] = [
                result.vector,
                result.direction,
                result.length,
                result.position,
                result.duration,
            ]
            results[j, i] = results[i, j]

    return results
