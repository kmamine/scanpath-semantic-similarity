"""
NLP-based similarity metrics for scanpath descriptions.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def rouge_score(text_a: str, text_b: str) -> float:
    """Compute ROUGE-L F1 score between two texts."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(text_a, text_b)
    return scores["rougeL"].fmeasure


def bleu_score(
    text_a: str,
    text_b: str,
    max_n: int = 4,
    weights: Optional[Tuple[float, ...]] = None,
) -> float:
    """Compute BLEU score between two texts."""
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize

    if weights is None:
        weights = tuple([1.0 / max_n] * max_n)

    ref_tokens = [word_tokenize(text_a.lower())]
    hyp_tokens = word_tokenize(text_b.lower())

    if len(hyp_tokens) == 0 or len(ref_tokens[0]) == 0:
        return 0.0

    return sentence_bleu(
        ref_tokens,
        hyp_tokens,
        weights=weights,
    )


def bertscore(
    text_a: str,
    text_b: str,
    model_type: str = "roberta-large",
    lang: str = "en",
    rescale_with_baseline: bool = True,
) -> float:
    """Compute BERTScore F1 between two texts."""
    import bert_score

    P, R, F1 = bert_score.score(
        [text_a],
        [text_b],
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        verbose=False,
    )
    return float(F1[0])


def compute_pairwise_rouge(
    descriptions: List[str],
) -> np.ndarray:
    """Compute pairwise ROUGE-L scores for a list of descriptions."""
    n = len(descriptions)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            score = rouge_score(descriptions[i], descriptions[j])
            matrix[i, j] = score
            matrix[j, i] = score

    np.fill_diagonal(matrix, 1.0)
    return matrix


def compute_pairwise_bleu(
    descriptions: List[str],
    max_n: int = 4,
) -> np.ndarray:
    """Compute pairwise BLEU scores."""
    n = len(descriptions)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            score_ij = bleu_score(descriptions[i], descriptions[j], max_n)
            score_ji = bleu_score(descriptions[j], descriptions[i], max_n)
            score = (score_ij + score_ji) / 2
            matrix[i, j] = score
            matrix[j, i] = score

    np.fill_diagonal(matrix, 1.0)
    return matrix


def compute_pairwise_bertscore(
    descriptions: List[str],
    model_type: str = "roberta-large",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute pairwise BERTScore using batched computation."""
    import bert_score

    n = len(descriptions)

    all_preds = []
    all_refs = []

    for i in range(n):
        for j in range(i + 1, n):
            all_preds.append(descriptions[i])
            all_refs.append(descriptions[j])

    if len(all_preds) == 0:
        matrix = np.ones((n, n), dtype=np.float32)
        return matrix

    _, _, F1 = bert_score.score(
        all_preds,
        all_refs,
        model_type=model_type,
        lang="en",
        rescale_with_baseline=True,
        batch_size=batch_size,
        verbose=False,
    )

    f1_scores = F1.numpy()

    matrix = np.zeros((n, n), dtype=np.float32)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = f1_scores[idx]
            matrix[j, i] = f1_scores[idx]
            idx += 1

    np.fill_diagonal(matrix, 1.0)
    return matrix


class BM25Scorer:
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


def compute_pairwise_bm25(
    descriptions: List[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    """Compute pairwise BM25 scores."""
    n = len(descriptions)
    scorer = BM25Scorer(descriptions, k1=k1, b=b)

    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            score = scorer.score_pair(descriptions[i], descriptions[j])
            matrix[i, j] = score
            matrix[j, i] = score

    np.fill_diagonal(matrix, 1.0)
    return matrix
