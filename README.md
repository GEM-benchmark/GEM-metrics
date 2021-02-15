GEM-metrics
===========
Automatic metrics for GEM benchmark tasks. Can also be used standalone for evaluation of various natural 
language generation tasks.

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

To compute all default metrics for a file, run:
```
./run_metrics.py [-r references.json] outputs.json
```

See [`test_data`](test_data/) for example JSON file formats.

Use `./run_metrics.py -h` to see all available options.

By default, the “heavy” metrics (BERTScore, BLEURT and SAFEval) aren't computed. Use `--heavy-metrics` to compute them.


License
-------
Licensed under [the MIT license](LICENSE).
