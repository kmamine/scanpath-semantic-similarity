#!/usr/bin/env python
"""
Example script demonstrating scanpath_nlp_metrics usage.
"""

from scanpath_nlp_metrics import ScanpathComparator

# Create comparator with patch method
comparator = ScanpathComparator(
    vlm_base_url="http://localhost:8000/v1",
    vlm_api_key="x",
    method="patch",
    patch_size=96,
    screen_size=(1680, 1050),
)

# Sample scanpaths (x, y, duration in seconds)
scanpath_a = [
    [100, 200, 0.3],
    [150, 250, 0.2],
    [200, 300, 0.25],
]

scanpath_b = [
    [105, 205, 0.3],
    [155, 255, 0.2],
    [205, 305, 0.25],
]

# Note: This will fail without a running VLM server
# result = comparator.compare(scanpath_a, scanpath_b, "image.jpg")

print("ScanpathComparator initialized successfully!")
print(f"Available metrics: {comparator.metrics}")
