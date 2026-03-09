Examples
=========

Basic Usage
-----------

Compare two scanpaths on an image:

.. code-block:: python

    from scanpath_nlp_metrics import ScanpathComparator
    from PIL import Image

    # Initialize comparator
    comparator = ScanpathComparator(
        vlm_base_url="http://localhost:8000/v1",
        vlm_api_key="your-key",
        vlm_model="qwen2-vl-7b-instruct",
        method="patch",
        patch_size=96,
        screen_size=(1680, 1050),
    )

    # Define scanpaths as [[x, y, duration], ...]
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

    # Compare
    result = comparator.compare(scanpath_a, scanpath_b, "image.jpg")

    print(result["description_a"])
    print(result["description_b"])
    print(f"ROUGE: {result['rouge']}")
    print(f"ScanMatch: {result['scanmatch']}")

Using Marker Method
-------------------

Instead of patches, use circular markers around fixations:

.. code-block:: python

    comparator = ScanpathComparator(
        method="marker",
        marker_radius=100,
    )

    result = comparator.compare(scanpath_a, scanpath_b, "image.jpg")

Selecting Specific Metrics
--------------------------

Compute only a subset of metrics:

.. code-block:: python

    comparator = ScanpathComparator(
        metrics=["rouge", "bleu", "scanmatch", "dtw"],
    )

    result = comparator.compare(scanpath_a, scanpath_b, "image.jpg")

Using PIL Image
---------------

Pass a PIL Image directly instead of a path:

.. code-block:: python

    from PIL import Image

    img = Image.open("image.jpg")
    result = comparator.compare(scanpath_a, scanpath_b, img)

Return Value
------------

The ``compare()`` method returns a dictionary with:

- ``description_a``: VLM-generated description for scanpath A
- ``description_b``: VLM-generated description for scanpath B
- ``rouge``: ROUGE-L F1 score
- ``bleu``: BLEU score
- ``bert_score``: BERTScore F1
- ``bm25``: BM25 similarity
- ``scanmatch``: ScanMatch similarity (0-1)
- ``dtw``: DTW distance (normalized)
- ``hausdorff``: Hausdorff distance (normalized)
- ``levenshtein``: Levenshtein distance (normalized)
- ``tde``: TDE distance (normalized)
- ``multimatch_vector``: MultiMatch vector similarity
- ``multimatch_direction``: MultiMatch direction similarity
- ``multimatch_length``: MultiMatch length similarity
- ``multimatch_position``: MultiMatch position similarity
- ``multimatch_duration``: MultiMatch duration similarity
- ``multimatch_mean``: Average of all MultiMatch dimensions
