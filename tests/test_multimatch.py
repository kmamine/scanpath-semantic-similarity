"""
Unit tests for MultiMatch metrics.
"""

import pytest
import numpy as np
from scanpath_nlp_metrics.metrics.multimatch import compute_multimatch, MultiMatchResult


class TestMultiMatch:
    def test_identical_scanpaths(self):
        fixations = np.array(
            [(100, 200, 0.3), (150, 250, 0.2), (200, 300, 0.25)],
            dtype=[("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")],
        )
        result = compute_multimatch(fixations, fixations, screensize=[1680, 1050])

        assert result.vector == 1.0
        assert result.direction == 1.0
        assert result.length == 1.0
        assert result.position == 1.0
        assert 0.0 <= result.duration <= 1.0

    def test_similar_scanpaths(self):
        """Test that 2-fixation scanpaths return NaN (minimum 3 required)."""
        fixations_a = np.array(
            [(100, 200, 0.3), (150, 250, 0.2)],
            dtype=[("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")],
        )
        fixations_b = np.array(
            [(105, 205, 0.3), (155, 255, 0.2)],
            dtype=[("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")],
        )
        result = compute_multimatch(fixations_a, fixations_b, screensize=[1680, 1050])

        assert np.isnan(result.vector)

    def test_different_scanpaths(self):
        """Test that 2-fixation scanpaths return NaN (minimum 3 required)."""
        fixations_a = np.array(
            [(100, 100, 0.3), (200, 200, 0.2)],
            dtype=[("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")],
        )
        fixations_b = np.array(
            [(1000, 1000, 0.3), (1100, 1100, 0.2)],
            dtype=[("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")],
        )
        result = compute_multimatch(fixations_a, fixations_b, screensize=[1680, 1050])

        assert np.isnan(result.vector)

    def test_short_scanpaths(self):
        """Test that very short scanpaths (less than 3 fixations) return NaN."""
        fixations_a = np.array(
            [(100, 200, 0.3)],
            dtype=[("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")],
        )
        fixations_b = np.array(
            [(150, 250, 0.2)],
            dtype=[("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")],
        )
        result = compute_multimatch(fixations_a, fixations_b, screensize=[1680, 1050])

        assert np.isnan(result.vector)
        assert np.isnan(result.direction)


class TestMultiMatchResult:
    def test_to_dict(self):
        result = MultiMatchResult(
            vector=0.8, direction=0.7, length=0.9, position=0.85, duration=0.6
        )
        d = result.to_dict()

        assert d["vector"] == 0.8
        assert d["direction"] == 0.7
        assert d["length"] == 0.9
        assert d["position"] == 0.85
        assert d["duration"] == 0.6
