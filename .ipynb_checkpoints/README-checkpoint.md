# scanpath_nlp_metrics

NLP and spatial metrics for scanpath comparison using Vision-Language Models.

## Installation

```bash
pip install scanpath_nlp_metrics
```

## Quick Start

```python
from scanpath_nlp_metrics import ScanpathEvaluator

evaluator = ScanpathEvaluator()
results = evaluator.evaluate(scanpaths, descriptions)
```

## CLI Usage

```bash
# Generate scanpath descriptions
scanpath-generate --json_path data/scanpaths.json --imgs_dir data/imgs --output_dir outputs

# Compute metrics
scanpath-metrics --json_path data/scanpaths.json --imgs_dir data/imgs --descriptions outputs/descriptions.jsonl --output_dir outputs

# Analyze results
scanpath-analyze --metrics outputs/metrics.parquet --output_dir outputs/analysis
```

## License

AGPL-2.0 - See LICENSE file for details.
