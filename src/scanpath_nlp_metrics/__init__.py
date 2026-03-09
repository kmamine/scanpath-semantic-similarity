"""
scanpath_nlp_metrics - NLP and spatial metrics for scanpath comparison using Vision-Language Models.

This library provides tools for:
- Generating scanpath descriptions using VLMs
- Computing pairwise NLP metrics (ROUGE, BLEU, BERTScore, BM25)
- Computing spatial similarity metrics (MultiMatch, DTW, Hausdorff, etc.)
- Analyzing correlations and divergence between scanpaths
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import logging

from .config import Config, DatasetConfig, VLMConfig, MarkerConfig, ScanMatchConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class ScanpathEvaluator:
    """
    Main evaluator class for scanpath comparison.

    Provides a simple interface to compute pairwise metrics between scanpaths
    using NLP and spatial similarity measures.

    Usage:
        evaluator = ScanpathEvaluator()
        results = evaluator.evaluate(scanpaths, descriptions)
    """

    config: Config = field(default_factory=Config)
    compute_multimatch: bool = True

    def __post_init__(self):
        self._comparator = None

    def set_vlm_config(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "x",
        model: str = "qwen2-vl-7b-instruct",
    ):
        """Configure VLM settings."""
        self.config.vlm.base_url = base_url
        self.config.vlm.api_key = api_key
        self.config.vlm.model = model

    def set_marker_config(
        self,
        radius: int = 100,
        outline_width: int = 3,
        dot_radius: int = 5,
    ):
        """Configure marker settings for fixation visualization."""
        self.config.marker.radius = radius
        self.config.marker.outline_width = outline_width
        self.config.marker.dot_radius = dot_radius

    def set_dataset_config(
        self,
        json_path: Optional[Path] = None,
        imgs_dir: Optional[Path] = None,
    ):
        """Configure dataset paths."""
        if json_path:
            self.config.dataset.json_path = json_path
        if imgs_dir:
            self.config.dataset.imgs_dir = imgs_dir

    def evaluate(
        self,
        scanpaths: Dict[str, Any],
        descriptions: Dict[str, str],
        compute_multimatch: Optional[bool] = None,
    ) -> Any:
        """
        Compute pairwise metrics for all scanpaths.

        Args:
            scanpaths: Dictionary mapping image_id to scanpath data
            descriptions: Dictionary mapping image_id to description text
            compute_multimatch: Whether to compute MultiMatch metrics

        Returns:
            DataFrame with pairwise comparison results
        """
        from .comparison.pairwise import PairwiseComparator

        compute_mm = (
            compute_multimatch
            if compute_multimatch is not None
            else self.compute_multimatch
        )

        comparator = PairwiseComparator(
            config=self.config,
            compute_multimatch=compute_mm,
        )

        results_df = comparator.compute_all(scanpaths, descriptions)
        return results_df

    def analyze_correlations(
        self,
        metrics_df: Any,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Analyze correlations between metrics.

        Args:
            metrics_df: DataFrame with metrics results
            output_dir: Directory to save correlation plots

        Returns:
            Dictionary with correlation results
        """
        from .analysis.correlation import generate_correlation_report

        if output_dir is None:
            output_dir = Path("outputs/correlation")

        return generate_correlation_report(metrics_df, output_dir)

    def find_divergent_cases(
        self,
        metrics_df: Any,
        descriptions: Optional[Dict[str, str]] = None,
        nlp_metric: str = "bertscore",
        n_examples: int = 20,
    ) -> Dict[str, Any]:
        """
        Find divergent and convergent scanpath pairs.

        Args:
            metrics_df: DataFrame with metrics results
            descriptions: Optional descriptions for enrichment
            nlp_metric: Primary NLP metric for divergence analysis
            n_examples: Number of examples to return

        Returns:
            Dictionary with divergence analysis results
        """
        from .analysis.divergence import generate_divergence_report

        output_dir = Path("outputs/divergence")
        return generate_divergence_report(
            metrics_df,
            output_dir,
            descriptions=descriptions,
            nlp_metric=nlp_metric,
            n_examples=n_examples,
        )


__version__ = "0.0.1"

__all__ = [
    "ScanpathEvaluator",
    "Config",
    "DatasetConfig",
    "VLMConfig",
    "MarkerConfig",
    "ScanMatchConfig",
    "log",
]
