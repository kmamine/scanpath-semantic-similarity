Installation
============

Install from PyPI
-----------------

.. code-block:: bash

    pip install scanpath-nlp-metrics

Install from source
-------------------

.. code-block:: bash

    git clone https://github.com/yourusername/scanpath_nlp_metrics.git
    cd scanpath_nlp_metrics
    pip install -e .

Requirements
------------

- Python 3.9+
- numpy
- Pillow
- rouge-score
- nltk
- bert-score
- rank-bm25
- scipy
- openai

Optional dependencies for spatial metrics:

- fastdtw

VLM Setup
---------

The library requires a VLM API endpoint. By default, it connects to:

- URL: ``http://localhost:8000/v1``
- Model: ``qwen2-vl-7b-instruct``

You can use any OpenAI-compatible VLM server.
