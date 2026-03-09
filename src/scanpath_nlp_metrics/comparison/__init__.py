"""
Comparison module for pairwise scanpath metrics.
"""

from .pairwise import (
    PairwiseResult,
    PairwiseComparator,
    load_descriptions_from_jsonl,
)

__all__ = [
    "PairwiseResult",
    "PairwiseComparator",
    "load_descriptions_from_jsonl",
]
