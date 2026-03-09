"""
Divergence analysis: Find cases where NLP and spatial metrics disagree.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class DivergentCase:
    """Container for a divergent case (high NLP, low spatial)."""

    image_id: str
    pair_i: int
    pair_j: int

    nlp_metric: str
    nlp_score: float

    spatial_metric: str
    spatial_score: float

    divergence: float

    description_i: Optional[str] = None
    description_j: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def find_divergent_cases(
    df: pd.DataFrame,
    nlp_metric: str = "bertscore",
    spatial_metric: str = "scanmatch",
    nlp_threshold: float = 0.6,
    spatial_threshold: float = 0.4,
    n_examples: int = 20,
    sort_by: str = "divergence",
) -> List[DivergentCase]:
    """Find cases where NLP similarity is high but spatial similarity is low."""
    valid_mask = df[nlp_metric].notna() & df[spatial_metric].notna()
    filtered = df[valid_mask].copy()

    divergent_mask = (filtered[nlp_metric] >= nlp_threshold) & (
        filtered[spatial_metric] <= spatial_threshold
    )

    divergent_df = filtered[divergent_mask].copy()

    if len(divergent_df) == 0:
        log.warning("No divergent cases found with current thresholds")
        return []

    divergent_df["divergence"] = divergent_df[nlp_metric] - divergent_df[spatial_metric]

    if sort_by == "divergence":
        divergent_df = divergent_df.sort_values("divergence", ascending=False)
    elif sort_by == "nlp_score":
        divergent_df = divergent_df.sort_values(nlp_metric, ascending=False)
    elif sort_by == "spatial_score":
        divergent_df = divergent_df.sort_values(spatial_metric, ascending=True)

    cases = []
    for _, row in divergent_df.head(n_examples).iterrows():
        case = DivergentCase(
            image_id=row["image_id"],
            pair_i=int(row["pair_i"]),
            pair_j=int(row["pair_j"]),
            nlp_metric=nlp_metric,
            nlp_score=float(row[nlp_metric]),
            spatial_metric=spatial_metric,
            spatial_score=float(row[spatial_metric]),
            divergence=float(row["divergence"]),
        )
        cases.append(case)

    log.info(f"Found {len(cases)} divergent cases")
    return cases


def find_convergent_cases(
    df: pd.DataFrame,
    nlp_metric: str = "bertscore",
    spatial_metric: str = "scanmatch",
    min_threshold: float = 0.7,
    n_examples: int = 20,
) -> List[DivergentCase]:
    """Find cases where both NLP and spatial similarity are high."""
    valid_mask = df[nlp_metric].notna() & df[spatial_metric].notna()
    filtered = df[valid_mask].copy()

    convergent_mask = (filtered[nlp_metric] >= min_threshold) & (
        filtered[spatial_metric] >= min_threshold
    )

    convergent_df = filtered[convergent_mask].copy()
    convergent_df["divergence"] = (
        convergent_df[nlp_metric] - convergent_df[spatial_metric]
    )

    convergent_df = convergent_df.sort_values("divergence", key=abs)

    cases = []
    for _, row in convergent_df.head(n_examples).iterrows():
        case = DivergentCase(
            image_id=row["image_id"],
            pair_i=int(row["pair_i"]),
            pair_j=int(row["pair_j"]),
            nlp_metric=nlp_metric,
            nlp_score=float(row[nlp_metric]),
            spatial_metric=spatial_metric,
            spatial_score=float(row[spatial_metric]),
            divergence=float(row["divergence"]),
        )
        cases.append(case)

    log.info(f"Found {len(cases)} convergent cases")
    return cases


def enrich_divergent_cases_with_descriptions(
    cases: List[DivergentCase],
    descriptions: Dict[str, List[str]],
) -> List[DivergentCase]:
    """Add description text to divergent cases."""
    for case in cases:
        if case.image_id in descriptions:
            desc_list = descriptions[case.image_id]
            if case.pair_i < len(desc_list):
                case.description_i = desc_list[case.pair_i]
            if case.pair_j < len(desc_list):
                case.description_j = desc_list[case.pair_j]
    return cases


def save_divergent_cases(
    cases: List[DivergentCase],
    output_path: Path,
) -> None:
    """Save divergent cases to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [c.to_dict() for c in cases]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.info(f"Saved {len(cases)} divergent cases to {output_path}")


def load_divergent_cases(
    input_path: Path,
) -> List[DivergentCase]:
    """Load divergent cases from JSON file."""
    with open(input_path) as f:
        data = json.load(f)

    return [DivergentCase(**item) for item in data]


def analyze_divergence_distribution(
    df: pd.DataFrame,
    nlp_metric: str = "bertscore",
    spatial_metric: str = "scanmatch",
    n_bins: int = 10,
) -> Dict:
    """Analyze the distribution of divergence values."""
    valid_mask = df[nlp_metric].notna() & df[spatial_metric].notna()
    filtered = df[valid_mask].copy()

    divergence = filtered[nlp_metric] - filtered[spatial_metric]

    hist, bin_edges = np.histogram(divergence, bins=n_bins, range=(-1, 1))

    return {
        "histogram": hist.tolist(),
        "bin_edges": bin_edges.tolist(),
        "mean": float(divergence.mean()),
        "std": float(divergence.std()),
        "median": float(divergence.median()),
        "n_positive": int((divergence > 0).sum()),
        "n_negative": int((divergence < 0).sum()),
        "n_zero": int((divergence == 0).sum()),
    }


def generate_divergence_report(
    df: pd.DataFrame,
    output_dir: Path,
    descriptions: Optional[Dict[str, List[str]]] = None,
    nlp_metric: str = "bertscore",
    spatial_metrics: Optional[List[str]] = None,
    n_examples: int = 20,
) -> Dict[str, List[DivergentCase]]:
    """Generate complete divergence analysis report."""
    if spatial_metrics is None:
        spatial_metrics = ["scanmatch", "mm_average", "dtw"]

    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases: Dict[str, List[DivergentCase]] = {}

    for sp_metric in spatial_metrics:
        if sp_metric not in df.columns:
            log.warning(f"Spatial metric {sp_metric} not found in DataFrame")
            continue

        log.info(f"Finding divergent cases for {nlp_metric} vs {sp_metric}...")

        cases = find_divergent_cases(
            df,
            nlp_metric=nlp_metric,
            spatial_metric=sp_metric,
            n_examples=n_examples,
        )

        if descriptions:
            cases = enrich_divergent_cases_with_descriptions(cases, descriptions)

        save_divergent_cases(
            cases, output_dir / f"divergent_{nlp_metric}_vs_{sp_metric}.json"
        )
        all_cases[f"{nlp_metric}_vs_{sp_metric}"] = cases

    log.info("Finding convergent cases...")
    convergent = find_convergent_cases(df, nlp_metric=nlp_metric, n_examples=n_examples)
    if descriptions:
        convergent = enrich_divergent_cases_with_descriptions(convergent, descriptions)
    save_divergent_cases(convergent, output_dir / "convergent_cases.json")
    all_cases["convergent"] = convergent

    log.info("Analyzing divergence distribution...")
    dist_analysis = analyze_divergence_distribution(df, nlp_metric=nlp_metric)
    with open(output_dir / "divergence_distribution.json", "w") as f:
        json.dump(dist_analysis, f, indent=2)

    return all_cases
