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

Usage
-----

To compute all metrics, run:
```
./run_metrics.py [-r references.json] outputs.json
```

