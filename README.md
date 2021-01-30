GEM-metrics
===========
Automatic metrics for GEM tasks.

Installation
------------

Requires recent Python 3, virtualenv or similar is recommended. To install, simply run:
```
git clone https://github.com/GEM-benchmark/GEM-metrics
cd GEM-metrics
pip install -r requirements.txt
```

Note that some NLTK stuff may be downloaded into a subdirectory of your checkout, so make sure you have write access when you run this.

Usage
-----

To compute all metrics, run:
```
./run_metrics.py [-r references.json] outputs.json
```

Use `./run_metrics.py -h` to see all available options.

By default, the “heavy” metrics (BERTScore and BLEURT) aren't computed. Use `--heavy-metrics` to compute them.
