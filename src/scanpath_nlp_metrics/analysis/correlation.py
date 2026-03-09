"""
Correlation analysis between NLP and spatial metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

log = logging.getLogger(__name__)


METRIC_COLUMNS = [
    "rouge",
    "bleu",
    "bertscore",
    "bm25",
    "scanmatch",
    "dtw",
    "tde",
    "hausdorff",
    "levenshtein",
    "mm_vector",
    "mm_direction",
    "mm_length",
    "mm_position",
    "mm_duration",
    "mm_average",
]

NLP_METRICS = ["rouge", "bleu", "bertscore", "bm25"]
SPATIAL_METRICS = ["scanmatch", "dtw", "tde", "hausdorff", "levenshtein"]
MULTIMATCH_METRICS = [
    "mm_vector",
    "mm_direction",
    "mm_length",
    "mm_position",
    "mm_duration",
    "mm_average",
]


def compute_spearman_correlation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute pairwise Spearman correlation between metrics."""
    if columns is None:
        columns = [c for c in METRIC_COLUMNS if c in df.columns]

    n = len(columns)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(columns):
            valid_mask = df[col_i].notna() & df[col_j].notna()
            if valid_mask.sum() > 2:
                corr, pval = spearmanr(
                    df.loc[valid_mask, col_i], df.loc[valid_mask, col_j]
                )
                corr_matrix[i, j] = corr
                p_matrix[i, j] = pval
            else:
                corr_matrix[i, j] = np.nan
                p_matrix[i, j] = np.nan

    corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
    p_df = pd.DataFrame(p_matrix, index=columns, columns=columns)

    return corr_df, p_df


def compute_pearson_correlation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise Pearson correlation between metrics."""
    if columns is None:
        columns = [c for c in METRIC_COLUMNS if c in df.columns]

    n = len(columns)
    corr_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(columns):
            valid_mask = df[col_i].notna() & df[col_j].notna()
            if valid_mask.sum() > 2:
                corr, pval = pearsonr(
                    df.loc[valid_mask, col_i], df.loc[valid_mask, col_j]
                )
                corr_matrix[i, j] = corr
                p_matrix[i, j] = pval
            else:
                corr_matrix[i, j] = np.nan
                p_matrix[i, j] = np.nan

    corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
    p_df = pd.DataFrame(p_matrix, index=columns, columns=columns)

    return corr_df, p_df


def compute_nlp_vs_spatial_correlations(
    df: pd.DataFrame,
    nlp_metrics: Optional[List[str]] = None,
    spatial_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute correlations between NLP metrics and spatial metrics."""
    if nlp_metrics is None:
        nlp_metrics = [c for c in NLP_METRICS if c in df.columns]
    if spatial_metrics is None:
        spatial_metrics = [
            c for c in SPATIAL_METRICS + MULTIMATCH_METRICS if c in df.columns
        ]

    corr_matrix = np.zeros((len(nlp_metrics), len(spatial_metrics)))
    p_matrix = np.zeros((len(nlp_metrics), len(spatial_metrics)))

    for i, nlp_col in enumerate(nlp_metrics):
        for j, sp_col in enumerate(spatial_metrics):
            valid_mask = df[nlp_col].notna() & df[sp_col].notna()
            if valid_mask.sum() > 2:
                corr, pval = spearmanr(
                    df.loc[valid_mask, nlp_col], df.loc[valid_mask, sp_col]
                )
                corr_matrix[i, j] = corr
                p_matrix[i, j] = pval
            else:
                corr_matrix[i, j] = np.nan
                p_matrix[i, j] = np.nan

    corr_df = pd.DataFrame(corr_matrix, index=nlp_metrics, columns=spatial_metrics)
    p_df = pd.DataFrame(p_matrix, index=nlp_metrics, columns=spatial_metrics)

    return corr_df, p_df


def save_correlation_results(
    corr_df: pd.DataFrame,
    p_df: pd.DataFrame,
    output_dir: Path,
    prefix: str = "correlation",
) -> None:
    """Save correlation results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    corr_df.to_csv(output_dir / f"{prefix}_matrix.csv")
    p_df.to_csv(output_dir / f"{prefix}_pvalues.csv")

    log.info(f"Saved correlation results to {output_dir}")


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    output_path: Path,
    title: str = "Metric Correlation Matrix",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = True,
) -> None:
    """Create and save correlation heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    mask = np.zeros_like(corr_df.values)
    mask[np.isnan(corr_df.values)] = True

    sns.heatmap(
        corr_df,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved correlation heatmap to {output_path}")


def generate_correlation_report(
    df: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Generate complete correlation analysis report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    log.info("Computing full Spearman correlation matrix...")
    full_corr, full_p = compute_spearman_correlation(df)
    results["full_correlation"] = full_corr
    results["full_pvalues"] = full_p
    save_correlation_results(full_corr, full_p, output_dir, "full_spearman")
    plot_correlation_heatmap(
        full_corr,
        output_dir / "full_correlation_heatmap.png",
        title="Spearman Correlation Between All Metrics",
    )

    log.info("Computing NLP vs Spatial correlations...")
    nlp_sp_corr, nlp_sp_p = compute_nlp_vs_spatial_correlations(df)
    results["nlp_vs_spatial"] = nlp_sp_corr
    results["nlp_vs_spatial_pvalues"] = nlp_sp_p
    save_correlation_results(nlp_sp_corr, nlp_sp_p, output_dir, "nlp_vs_spatial")
    plot_correlation_heatmap(
        nlp_sp_corr,
        output_dir / "nlp_vs_spatial_heatmap.png",
        title="NLP vs Spatial Metric Correlations",
        figsize=(10, 6),
    )

    return results
