GEM-metrics
===========
Automatic metrics for GEM benchmark tasks. Can also be used standalone for evaluation of various natural 
language generation tasks.

Installation
------------

GEM-metrics require recent Python 3, virtualenv or similar is recommended. To install, simply run:
```
git clone https://github.com/GEM-benchmark/GEM-metrics
cd GEM-metrics
pip install -r requirements.txt -r requirements-heavy.txt
```

If you want to just run the metrics from console (and don't need direct access to the source code), you can just run:
```
pip install 'gem-metrics[heavy] @ git+https://github.com/GEM-benchmark/GEM-metrics.git'
```

Note that some NLTK stuff may be downloaded upon first run into a subdirectory where the code is located, 
so make sure you have write access when you run this.
Also note that all the required Python libraries are around 3 GB in size when installed.

If you don't need trained metrics (BLEURT, BERTScore, NUBIA, QuestEval), you can ignore the “heavy” part, 
i.e. only install dependencies from `requirements.txt` or only use `gem-metrics` instead of `gem-metrics[heavy]`
if installing without checkout. That way, your installed libraries will be ~300 MB.

Script Usage
------------

To compute all default metrics for a file, run:
```
<script> [-r references.json] outputs.json
```
Where `<script>` is either `./run_metrics.py` (if you created a checkout) or `gem_metrics` if you installed directly via `pip`.

See [`test_data`](test_data/) for example JSON file formats.

For calculating basic metrics with the unit test data, run:
```
./run_metrics.py -s test_data/unit_tests/sources.json  -r test_data/unit_tests/references.json test_data/unit_tests/predictions.json
```

Use `./run_metrics.py -h` to see all available options.

By default, the “heavy” metrics (BERTScore, BLEURT, NUBIA and QuestEval) aren't computed. Use `--heavy-metrics` to compute them.


Library Usage
-------------

You can compute metrics for the same JSON format as shown in [`test_data`](test_data/), or you can work 
with plain lists of texts (or lists of lists of texts in the case of multi-reference data).

Import GEM-metrics as a library:
```
import gem_metrics
```

To load data from JSON files:
```
preds = gem_metrics.texts.Predictions('path/to/pred-file.json')
refs = gem_metrics.texts.References('path/to/ref-file.json')
```

To prepare plain lists (assuming the same order):
```
preds = gem_metrics.texts.Predictions(list_of_predictions)
refs = gem_metrics.texts.References(list_of_references)  # input may be list of lists for multi-ref
```

Then compute the desired metrics:
```
result = gem_metrics.compute(preds, refs, metrics_list=['bleu', 'rouge'])  # add list of desired metrics here
```


List of supported metrics
-------------------------

Referenceless:

* `local_recall` -- LocalRecall
* `msttr` -- MSTTR
* `ngrams` -- n-gram statistics
* `ttr` -- TTR


Reference-based:

* `bertscore` -- BERTScore (heavy)
* `bleu` -- BLEU
* `bleurt` -- BLEURT (heavy)
* `chrf` -- CHRF
* `cider` -- CIDER
* `meteor` -- Meteor (heavy)
* `moverscore` -- MoverScore (heavy)
* `nist` -- NIST
* `nubia` -- NUBIA (heavy)
* `prism` -- Prism
* `questeval` -- QuestEval (heavy)
* `rouge` -- ROUGE
* `ter` -- TER
* `wer` -- WER
* `yules_i` -- Yules_I

Source + reference based:

* `sari` -- SARI

License
-------
Licensed under [the MIT license](LICENSE).
