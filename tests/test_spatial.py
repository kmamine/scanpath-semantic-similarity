"""
Unit tests for spatial metrics.
"""

import pytest
import numpy as np
from scanpath_nlp_metrics.metrics.spatial_metrics import (
    dtw_distance,
    hausdorff_distance,
    levenshtein_distance,
    tde_distance,
    normalize_distance,
    ScanMatch,
)


class TestDTW:
    def test_identical_scanpaths(self):
        scanpath = np.array([[100, 200], [150, 250], [200, 300]])
        score = dtw_distance(scanpath, scanpath)
        assert score == 0.0

    def test_similar_scanpaths(self):
        scanpath_a = np.array([[100, 200], [150, 250], [200, 300]])
        scanpath_b = np.array([[105, 205], [155, 255], [205, 305]])
        score = dtw_distance(scanpath_a, scanpath_b)
        assert score >= 0.0

    def test_different_scanpaths(self):
        scanpath_a = np.array([[0, 0], [100, 100], [200, 200]])
        scanpath_b = np.array([[1000, 1000], [1100, 1100], [1200, 1200]])
        score = dtw_distance(scanpath_a, scanpath_b)
        assert score > 0.0


class TestHausdorff:
    def test_identical_scanpaths(self):
        scanpath = np.array([[100, 200], [150, 250], [200, 300]])
        score = hausdorff_distance(scanpath, scanpath)
        assert score == 0.0

    def test_similar_scanpaths(self):
        scanpath_a = np.array([[100, 200], [150, 250]])
        scanpath_b = np.array([[105, 205], [155, 255]])
        score = hausdorff_distance(scanpath_a, scanpath_b)
        assert score >= 0.0


class TestLevenshtein:
    def test_identical_scanpaths(self):
        scanpath = np.array([[100, 200, 0.3], [150, 250, 0.2], [200, 300, 0.25]])
        score = levenshtein_distance(scanpath, scanpath, height=1050, width=1680)
        assert score == 0.0

    def test_different_scanpaths(self):
        scanpath_a = np.array([[0, 0, 0.1], [100, 100, 0.1]])
        scanpath_b = np.array([[500, 500, 0.1], [600, 600, 0.1]])
        score = levenshtein_distance(scanpath_a, scanpath_b, height=1050, width=1680)
        assert score > 0.0


class TestTDE:
    def test_identical_scanpaths(self):
        scanpath = np.array([[100, 200], [150, 250], [200, 300], [250, 350]])
        score = tde_distance(scanpath, scanpath, k=3)
        assert score == 0.0

    def test_similar_scanpaths(self):
        scanpath_a = np.array([[100, 200], [150, 250], [200, 300], [250, 350]])
        scanpath_b = np.array([[105, 205], [155, 255], [205, 305], [255, 355]])
        score = tde_distance(scanpath_a, scanpath_b, k=3)
        assert score >= 0.0


class TestNormalizeDistance:
    def test_zero_distance(self):
        assert normalize_distance(0.0) == 1.0

    def test_large_distance(self):
        score = normalize_distance(5000.0)
        assert 0.0 <= score <= 1.0

    def test_custom_max(self):
        score = normalize_distance(50.0, max_distance=100.0)
        assert score == 0.5


class TestScanMatch:
    def test_identical_scanpaths(self):
        sm = ScanMatch(xres=1680, yres=1050)
        scanpath = np.array([[100, 200, 0.3], [150, 250, 0.2]])
        score = sm.score(scanpath, scanpath)
        assert score == 1.0

    def test_similar_scanpaths(self):
        sm = ScanMatch(xres=1680, yres=1050)
        scanpath_a = np.array([[100, 200, 0.3], [150, 250, 0.2]])
        scanpath_b = np.array([[105, 205, 0.3], [155, 255, 0.2]])
        score = sm.score(scanpath_a, scanpath_b)
        assert 0.0 <= score <= 1.0

    def test_different_scanpaths(self):
        sm = ScanMatch(xres=1680, yres=1050)
        scanpath_a = np.array([[100, 100, 0.3], [200, 200, 0.2]])
        scanpath_b = np.array([[1000, 1000, 0.3], [1100, 1100, 0.2]])
        score = sm.score(scanpath_a, scanpath_b)
        assert 0.0 <= score <= 1.0
