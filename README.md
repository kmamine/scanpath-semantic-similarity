# scanpath_nlp_metrics

NLP and spatial metrics for scanpath comparison using Vision-Language Models.

## Installation

```bash
pip install scanpath_nlp_metrics
```

## Quick Start

```python
from scanpath_nlp_metrics import ScanpathComparator

comparator = ScanpathComparator(
    vlm_base_url="http://localhost:8000/v1",
    vlm_api_key="your-key",
    vlm_model="qwen2-vl-7b-instruct",
    method="patch",       # or "marker"
    patch_size=96,        # for patch method
    marker_radius=100,     # for marker method
)

result = comparator.compare(
    scanpath_a=[[100, 200, 0.3], [150, 250, 0.2], ...],
    scanpath_b=[[120, 210, 0.25], [160, 240, 0.15], ...],
    image="path/to/image.jpg",
)
```

## Output

```python
{
    "description_a": "The viewer first looked at the person's face...",
    "description_b": "The viewer focused on the main subject...",
    "rouge": 0.45,
    "bleu": 0.32,
    "bert_score": 0.67,
    "bm25": 0.55,
    "scanmatch": 0.78,
    "dtw": 0.82,
    "hausdorff": 0.65,
    "levenshtein": 0.70,
    "tde": 0.75,
    "multimatch_vector": 0.65,
    "multimatch_direction": 0.71,
    "multimatch_length": 0.68,
    "multimatch_position": 0.72,
    "multimatch_duration": 0.55,
    "multimatch_mean": 0.66,
}
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vlm_base_url` | `http://localhost:8000/v1` | VLM API endpoint |
| `vlm_api_key` | `x` | API key |
| `vlm_model` | `qwen2-vl-7b-instruct` | Model name |
| `method` | `patch` | Generation method: `patch` or `marker` |
| `patch_size` | `96` | Patch size in pixels |
| `marker_radius` | `100` | Marker radius in pixels |
| `screen_size` | `(1680, 1050)` | Screen resolution |
| `metrics` | all | List of metrics to compute |

## Available Metrics

- **NLP**: `rouge`, `bleu`, `bert_score`, `bm25`
- **Spatial**: `scanmatch`, `dtw`, `hausdorff`, `levenshtein`, `tde`
- **MultiMatch**: `multimatch_vector`, `multimatch_direction`, `multimatch_length`, `multimatch_position`, `multimatch_duration`, `multimatch_mean`

## License

AGPL-2.0 - See LICENSE file for details.
