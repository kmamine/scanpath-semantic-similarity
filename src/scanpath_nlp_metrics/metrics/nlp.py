"""
NLP-based similarity metrics for scanpath descriptions.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def rouge(text_a: str, text_b: str) -> float:
    """Compute ROUGE-L F1 score between two texts."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(text_a, text_b)
    return scores["rougeL"].fmeasure


def bleu(text_a: str, text_b: str, max_n: int = 4) -> float:
    """Compute BLEU score between two texts."""
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize

    weights = tuple([1.0 / max_n] * max_n)

    ref_tokens = [word_tokenize(text_a.lower())]
    hyp_tokens = word_tokenize(text_b.lower())

    if len(hyp_tokens) == 0 or len(ref_tokens[0]) == 0:
        return 0.0

    return sentence_bleu(ref_tokens, hyp_tokens, weights=weights)


def bert_score(text_a: str, text_b: str, model_type: str = "roberta-large") -> float:
    """Compute BERTScore F1 between two texts."""
    import bert_score

    _, _, F1 = bert_score.score(
        [text_a],
        [text_b],
        model_type=model_type,
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
    )
    return float(F1[0])


class BM25:
    """BM25 scorer for computing lexical similarity with IDF weighting."""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        from rank_bm25 import BM25Okapi
        from nltk.tokenize import word_tokenize

        self.corpus = corpus
        self.tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)
        self.tokenizer = word_tokenize

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score between query and a specific document."""
        query_tokens = self.tokenizer(query.lower())
        scores = self.bm25.get_scores(query_tokens)
        return float(scores[doc_idx])

    def score_pair(self, text_a: str, text_b: str) -> float:
        """Compute symmetric BM25 score between two texts."""
        token_a = self.tokenizer(text_a.lower())
        token_b = self.tokenizer(text_b.lower())

        scores_ab = self.bm25.get_scores(token_a)
        scores_ba = self.bm25.get_scores(token_b)

        idx_a = self._find_doc_index(token_a)
        idx_b = self._find_doc_index(token_b)

        if idx_a is None or idx_b is None:
            return 0.0

        score_a_to_b = float(scores_ab[idx_b])
        score_b_to_a = float(scores_ba[idx_a])

        max_score = max(score_a_to_b, score_b_to_a, 1e-6)
        return (score_a_to_b + score_b_to_a) / 2 / max_score

    def _find_doc_index(self, tokens: List[str]) -> Optional[int]:
        """Find document index by token match."""
        for i, doc_tokens in enumerate(self.tokenized_corpus):
            if doc_tokens == tokens:
                return i
        return None


def bm25(text_a: str, text_b: str, corpus: Optional[List[str]] = None) -> float:
    """Compute BM25 score between two texts."""
    if corpus is None:
        corpus = [text_a, text_b]

    scorer = BM25(corpus)
    return scorer.score_pair(text_a, text_b)


def compute_nlp_metrics(
    description_a: str,
    description_b: str,
    metrics: Optional[List[str]] = None,
) -> dict:
    """
    Compute all NLP metrics between two descriptions.

    Args:
        description_a: First description text
        description_b: Second description text
        metrics: List of metrics to compute (default: all)

    Returns:
        Dict with metric names as keys and scores as values
    """
    all_metrics = ["rouge", "bleu", "bert_score", "bm25"]

    if metrics is None:
        metrics = all_metrics
    else:
        metrics = [m for m in metrics if m in all_metrics]

    results = {}
    corpus = [description_a, description_b]

    if "rouge" in metrics:
        results["rouge"] = rouge(description_a, description_b)

    if "bleu" in metrics:
        results["bleu"] = bleu(description_a, description_b)

    if "bert_score" in metrics:
        results["bert_score"] = bert_score(description_a, description_b)

    if "bm25" in metrics:
        results["bm25"] = bm25(description_a, description_b, corpus)

    return results
