"""
Analysis module for correlation and divergence detection.
"""

from .correlation import (
    compute_spearman_correlation,
    compute_pearson_correlation,
    compute_nlp_vs_spatial_correlations,
    save_correlation_results,
    plot_correlation_heatmap,
    generate_correlation_report,
    METRIC_COLUMNS,
    NLP_METRICS,
    SPATIAL_METRICS,
    MULTIMATCH_METRICS,
)
from .divergence import (
    DivergentCase,
    find_divergent_cases,
    find_convergent_cases,
    enrich_divergent_cases_with_descriptions,
    save_divergent_cases,
    load_divergent_cases,
    analyze_divergence_distribution,
    generate_divergence_report,
)

__all__ = [
    "compute_spearman_correlation",
    "compute_pearson_correlation",
    "compute_nlp_vs_spatial_correlations",
    "save_correlation_results",
    "plot_correlation_heatmap",
    "generate_correlation_report",
    "METRIC_COLUMNS",
    "NLP_METRICS",
    "SPATIAL_METRICS",
    "MULTIMATCH_METRICS",
    "DivergentCase",
    "find_divergent_cases",
    "find_convergent_cases",
    "enrich_divergent_cases_with_descriptions",
    "save_divergent_cases",
    "load_divergent_cases",
    "analyze_divergence_distribution",
    "generate_divergence_report",
]
