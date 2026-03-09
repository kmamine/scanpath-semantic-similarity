#!/usr/bin/env python
"""
Example: Compute spatial and NLP metrics without VLM.

This shows how to use the individual metric functions directly.
"""

import numpy as np
from scanpath_nlp_metrics.metrics.spatial_metrics import (
    ScanMatch,
    dtw_distance,
    hausdorff_distance,
    levenshtein_distance,
    tde_distance,
    normalize_distance,
)
from scanpath_nlp_metrics.metrics.nlp import rouge, bleu, bert_score, bm25

# Sample scanpaths
scanpath_a = np.array([[100, 200, 0.3], [150, 250, 0.2], [200, 300, 0.25]])
scanpath_b = np.array([[105, 205, 0.3], [155, 255, 0.2], [205, 305, 0.25]])

# Screen size
screen_size = (1680, 1050)

# Compute spatial metrics
print("=== Spatial Metrics ===")

scanmatch = ScanMatch(xres=screen_size[0], yres=screen_size[1])
scanmatch_score = scanmatch.score(scanpath_a, scanpath_b)
print(f"ScanMatch: {scanmatch_score:.3f}")

dtw_dist = dtw_distance(scanpath_a[:, :2], scanpath_b[:, :2])
print(f"DTW: {normalize_distance(dtw_dist):.3f}")

haus_dist = hausdorff_distance(scanpath_a[:, :2], scanpath_b[:, :2])
print(f"Hausdorff: {normalize_distance(haus_dist):.3f}")

lev_dist = levenshtein_distance(
    scanpath_a, scanpath_b, height=screen_size[1], width=screen_size[0]
)
print(f"Levenshtein: {normalize_distance(lev_dist):.3f}")

tde_dist = tde_distance(scanpath_a[:, :2], scanpath_b[:, :2], k=3)
print(f"TDE: {normalize_distance(tde_dist):.3f}")

# Compute NLP metrics
print("\n=== NLP Metrics ===")

text_a = "The viewer first looked at the person's face, then moved to the background."
text_b = (
    "The viewer focused on the main subject's face before looking at the background."
)

r = rouge(text_a, text_b)
print(f"ROUGE: {r:.3f}")

b = bleu(text_a, text_b)
print(f"BLEU: {b:.3f}")

bs = bert_score(text_a, text_b)
print(f"BERTScore: {bs:.3f}")

bm = bm25(text_a, text_b)
print(f"BM25: {bm:.3f}")
