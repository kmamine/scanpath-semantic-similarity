"""
Unit tests for NLP metrics.
"""

import pytest
from scanpath_nlp_metrics.metrics.nlp import (
    rouge,
    bleu,
    bert_score,
    bm25,
    compute_nlp_metrics,
)


class TestROUGE:
    def test_identical_texts(self):
        text = "This is a test sentence"
        score = rouge(text, text)
        assert score == 1.0

    def test_completely_different(self):
        text_a = "The quick brown fox"
        text_b = "A lazy dog at night"
        score = rouge(text_a, text_b)
        assert 0.0 <= score <= 1.0

    def test_partial_overlap(self):
        text_a = "The quick brown fox jumps over"
        text_b = "The quick red fox runs past"
        score = rouge(text_a, text_b)
        assert 0.0 < score < 1.0


class TestBLEU:
    def test_identical_texts(self):
        text = "This is a test sentence"
        score = bleu(text, text)
        assert score == 1.0

    def test_different_texts(self):
        text_a = "The cat sat on the mat"
        text_b = "A dog ran in the park"
        score = bleu(text_a, text_b)
        assert 0.0 <= score <= 1.0


class TestBERTScore:
    def test_identical_texts(self):
        text = "The viewer looked at the person's face"
        score = bert_score(text, text)
        assert score > 0.9

    def test_similar_semantics(self):
        text_a = "A person looking at a face"
        text_b = "Someone staring at a person"
        score = bert_score(text_a, text_b)
        assert 0.0 <= score <= 1.0


class TestBM25:
    def test_identical_texts(self):
        text = "This is a test document"
        score = bm25(text, text)
        assert score == 1.0

    def test_different_texts(self):
        text_a = "The cat sat on the mat"
        text_b = "A dog ran in the park"
        score = bm25(text_a, text_b)
        assert 0.0 <= score <= 1.0


class TestComputeNLPMetrics:
    def test_all_metrics(self):
        text_a = "The viewer focused on the person's face"
        text_b = "The viewer looked at the person's face"
        results = compute_nlp_metrics(text_a, text_b)

        assert "rouge" in results
        assert "bleu" in results
        assert "bert_score" in results
        assert "bm25" in results

    def test_specific_metrics(self):
        text_a = "Test sentence one"
        text_b = "Test sentence two"
        results = compute_nlp_metrics(text_a, text_b, metrics=["rouge", "bleu"])

        assert "rouge" in results
        assert "bleu" in results
        assert "bert_score" not in results
        assert "bm25" not in results
