"""
Configuration for scanpath NLP metrics experiments.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfig:
    """Configuration for dataset paths and parameters."""

    json_path: Path = field(
        default_factory=lambda: Path("data/COCOFreeView_fixations_trainval.json")
    )
    imgs_dir: Path = field(default_factory=lambda: Path("data/imgs"))
    reduced_json: Path = field(
        default_factory=lambda: Path("data/reduced/COCOFreeView_reduced.json")
    )
    reduced_imgs_dir: Path = field(default_factory=lambda: Path("data/reduced/imgs"))

    n_images: int = 100
    n_scanpaths_per_image: int = 5
    split: str = "valid"
    seed: int = 42

    image_width: int = 1680
    image_height: int = 1050


@dataclass
class VLMConfig:
    """Configuration for Vision-Language Model."""

    base_url: str = "http://localhost:8000/v1"
    api_key: str = "x"
    model: str = "qwen2-vl-7b-instruct"

    patch_size: int = 96
    patch_max_tokens: int = 80
    scanpath_max_tokens: int = 180
    temperature_patch: float = 0.2
    temperature_scanpath: float = 0.3

    patch_prompt: str = (
        "Describe what you see in this image patch in 1-2 sentences. "
        "Focus on any objects, faces, text, or salient visual content. "
        "If the patch appears blurry or shows only texture/background, "
        "describe the dominant colour, texture, or any partial object visible."
    )

    scanpath_prompt_template: str = (
        "You are analysing where a human viewer looked at an imageBelow are sequential descriptions. "
        " of the image regions they fixated on (in temporal order):\n\n"
        "{fixation_list}\n\n"
        "Given the full image descriptions, write a provided and these fixation single coherent paragraph "
        "summarising what this viewer attended to and what cognitive strategy they might have used."
    )


@dataclass
class MarkerConfig:
    """Configuration for fixation marker visualization."""

    radius: int = 100
    outline_color: tuple = (255, 0, 0)
    outline_width: int = 3
    dot_radius: int = 5
    dot_color: tuple = (255, 0, 0)

    fixation_prompt: str = (
        "You are analyzing where a viewer looked at an image. "
        "The red circle marks the region they fixated on (the circle center is the exact gaze point).\n\n"
        "Describe what is inside the circled region in 1-2 sentences. Focus on:\n"
        "- Objects or elements within the circle\n"
        "- The visual content at the fixation location\n"
        "- How this region relates to the broader image context\n\n"
        "Be specific about what the viewer was looking at in that circled area."
    )


@dataclass
class ScanMatchConfig:
    """Configuration for MultiMatch scanpath comparison."""

    xres: int = 1680
    yres: int = 1050
    xbin: int = 14
    ybin: int = 9
    threshold: float = 3.5
    gap_value: float = 0.0
    temp_bin: float = 0.0
    offset: tuple = (0, 0)


@dataclass
class OutputConfig:
    """Configuration for output paths."""

    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    patch_descriptions_file: str = "patch_descriptions.jsonl"
    scanpath_descriptions_file: str = "scanpath_descriptions.jsonl"
    metrics_results_file: str = "metrics_results.parquet"
    correlation_matrix_file: str = "correlation_matrix.csv"
    divergent_cases_file: str = "divergent_cases.json"
    run_config_file: str = "run_config.json"


@dataclass
class Config:
    """Main configuration class."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    marker: MarkerConfig = field(default_factory=MarkerConfig)
    scanmatch: ScanMatchConfig = field(default_factory=ScanMatchConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Create Config from dictionary."""
        config = cls()
        if "dataset" in d:
            for k, v in d["dataset"].items():
                if hasattr(config.dataset, k):
                    setattr(
                        config.dataset, k, Path(v) if "path" in k or "dir" in k else v
                    )
        if "vlm" in d:
            for k, v in d["vlm"].items():
                if hasattr(config.vlm, k):
                    setattr(config.vlm, k, v)
        if "marker" in d:
            for k, v in d["marker"].items():
                if hasattr(config.marker, k):
                    setattr(config.marker, k, v)
        if "scanmatch" in d:
            for k, v in d["scanmatch"].items():
                if hasattr(config.scanmatch, k):
                    setattr(config.scanmatch, k, v)
        if "output" in d:
            for k, v in d["output"].items():
                if hasattr(config.output, k):
                    setattr(config.output, k, Path(v) if "dir" in k else v)
        return config
