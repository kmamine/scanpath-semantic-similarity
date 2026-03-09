"""
Scanpath description generation using full-image with fixation markers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from tqdm import tqdm

from ..config import Config
from ..data.loader import Scanpath, load_jsonl, append_jsonl
from ..data.image_marking import draw_fixation_marker
from .vlm_client import VLMClient

log = logging.getLogger(__name__)


@dataclass
class FixationDescription:
    image_id: str
    subject_id: int
    scanpath_idx: int
    fix_idx: int
    x: float
    y: float
    marker_radius: int
    description: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class ScanpathDescription:
    image_id: str
    subject_id: int
    scanpath_idx: int
    n_fixations: int
    fixation_descriptions: List[str]
    description: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


def make_fixation_key(
    image_id: str, subject_id: int, scanpath_idx: int, fix_idx: int, radius: int
) -> str:
    return f"{image_id}__{subject_id}__{scanpath_idx}__{fix_idx}__marker_{radius}"


def make_scanpath_key(image_id: str, subject_id: int, scanpath_idx: int) -> str:
    return f"{image_id}__{subject_id}__{scanpath_idx}"


class MarkerPipeline:
    """Generates descriptions using full images with fixation markers."""

    def __init__(
        self,
        config: Config,
        vlm_client: Optional[VLMClient] = None,
    ):
        self.config = config
        self.vlm = vlm_client or VLMClient(
            base_url=config.vlm.base_url,
            api_key=config.vlm.api_key,
            model=config.vlm.model,
        )
        self.fixation_cache: Dict[str, str] = {}
        self.scanpath_cache: Dict[str, str] = {}

    def load_cache(self, fixation_file: Path, scanpath_file: Path) -> None:
        """Load existing descriptions from JSONL files."""
        for record in load_jsonl(fixation_file):
            key = make_fixation_key(
                record["image_id"],
                record["subject_id"],
                record["scanpath_idx"],
                record["fix_idx"],
                record["marker_radius"],
            )
            self.fixation_cache[key] = record["description"]

        for record in load_jsonl(scanpath_file):
            key = make_scanpath_key(
                record["image_id"],
                record["subject_id"],
                record["scanpath_idx"],
            )
            self.scanpath_cache[key] = record["description"]

        log.info(
            f"Loaded cache: {len(self.fixation_cache)} fixations, "
            f"{len(self.scanpath_cache)} scanpaths"
        )

    def generate_for_scanpath(
        self,
        scanpath: Scanpath,
        image: Image.Image,
        scanpath_idx: int,
        fixation_out: Optional[Path] = None,
    ) -> List[str]:
        """Generate descriptions for all fixations in a scanpath."""
        fix_descriptions: List[str] = []

        for fix_idx, fixation in enumerate(scanpath.fixations):
            f_key = make_fixation_key(
                scanpath.image_id,
                scanpath.subject_id,
                scanpath_idx,
                fix_idx,
                self.config.marker.radius,
            )

            if f_key in self.fixation_cache:
                fix_descriptions.append(self.fixation_cache[f_key])
                continue

            marked_image = draw_fixation_marker(
                image,
                fixation.x,
                fixation.y,
                radius=self.config.marker.radius,
                outline_color=self.config.marker.outline_color,
                outline_width=self.config.marker.outline_width,
                dot_radius=self.config.marker.dot_radius,
                dot_color=self.config.marker.dot_color,
            )

            response = self.vlm.describe_patch(
                marked_image,
                self.config.marker.fixation_prompt,
                max_tokens=self.config.vlm.patch_max_tokens,
                temperature=self.config.vlm.temperature_patch,
            )

            fix_descriptions.append(response.content)
            self.fixation_cache[f_key] = response.content

            if fixation_out is not None:
                record = asdict(
                    FixationDescription(
                        image_id=scanpath.image_id,
                        subject_id=scanpath.subject_id,
                        scanpath_idx=scanpath_idx,
                        fix_idx=fix_idx,
                        x=fixation.x,
                        y=fixation.y,
                        marker_radius=self.config.marker.radius,
                        description=response.content,
                        model=self.config.vlm.model,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        latency_s=response.latency_s,
                    )
                )
                append_jsonl(fixation_out, record)

        return fix_descriptions

    def generate_scanpath_summary(
        self,
        scanpath: Scanpath,
        image: Image.Image,
        scanpath_idx: int,
        fix_descriptions: List[str],
        scanpath_out: Optional[Path] = None,
    ) -> str:
        """Generate summary description for a full scanpath."""
        sp_key = make_scanpath_key(
            scanpath.image_id,
            scanpath.subject_id,
            scanpath_idx,
        )

        if sp_key in self.scanpath_cache:
            return self.scanpath_cache[sp_key]

        response = self.vlm.describe_scanpath(
            image,
            fix_descriptions,
            self.config.vlm.scanpath_prompt_template,
            max_tokens=self.config.vlm.scanpath_max_tokens,
            temperature=self.config.vlm.temperature_scanpath,
        )

        self.scanpath_cache[sp_key] = response.content

        if scanpath_out is not None:
            record = asdict(
                ScanpathDescription(
                    image_id=scanpath.image_id,
                    subject_id=scanpath.subject_id,
                    scanpath_idx=scanpath_idx,
                    n_fixations=len(scanpath.fixations),
                    fixation_descriptions=fix_descriptions,
                    description=response.content,
                    model=self.config.vlm.model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    latency_s=response.latency_s,
                )
            )
            append_jsonl(scanpath_out, record)

        return response.content

    def run(
        self,
        scanpaths: Dict[str, List[Scanpath]],
        image_paths: Dict[str, Path],
        output_dir: Path,
    ) -> None:
        """Run the full description generation pipeline."""
        output_dir.mkdir(parents=True, exist_ok=True)

        fixation_out = output_dir / "fixation_descriptions.jsonl"
        sp_out = output_dir / "scanpath_descriptions.jsonl"
        config_out = output_dir / "run_config.json"

        self.load_cache(fixation_out, sp_out)

        with open(config_out, "w") as f:
            json.dump(
                {
                    "method": "full_image_marker",
                    "vlm": {
                        "base_url": self.config.vlm.base_url,
                        "model": self.config.vlm.model,
                    },
                    "marker": {
                        "radius": self.config.marker.radius,
                        "outline_color": self.config.marker.outline_color,
                        "outline_width": self.config.marker.outline_width,
                        "dot_radius": self.config.marker.dot_radius,
                    },
                    "dataset": {
                        "n_images": len(scanpaths),
                        "n_scanpaths": sum(len(sps) for sps in scanpaths.values()),
                    },
                },
                f,
                indent=2,
            )

        total_fix_calls = 0
        total_sp_calls = 0

        all_image_ids = list(scanpaths.keys())

        for image_id in tqdm(all_image_ids, desc="Images"):
            sp_list = scanpaths[image_id]
            img_path = image_paths[image_id]
            image = Image.open(img_path).convert("RGB")

            for sp_idx, sp in enumerate(sp_list):
                fix_descs = self.generate_for_scanpath(sp, image, sp_idx, fixation_out)
                total_fix_calls += len(sp.fixations)

                self.generate_scanpath_summary(sp, image, sp_idx, fix_descs, sp_out)
                total_sp_calls += 1

        log.info(
            f"Generation complete: {total_fix_calls} fixations, "
            f"{total_sp_calls} scanpaths"
        )
        log.info(f"Output directory: {output_dir}")
