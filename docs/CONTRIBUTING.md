# Implementing new metrics

By convention, new metrics are implemented in a separate file in the `gem_metrics` directory, such that BLEU, for example, is implemented in `gem_metrics/bleu.py`.

To implement a new metric, it is best to familiarize yourself with the abstract classes defined in `gem_metrics/metric.py` first, from which each metric is derived depending on whether it is reference-less, uses references, or references and sources.
Accordingly, every metric has to implement a `compute` function that computes the actual metric given either only predictions, predictions and references, or predictions, references and sources.
Furthermore, heavy metrics that are model-based potentially implement the `_initialize` function that can be used to load a model, for example.

Some examples:

1. Reference-less metric:
For an example, see: [MSTTR](https://github.com/GEM-benchmark/GEM-metrics/blob/main/gem_metrics/msttr.py)
    Reference-less metrics inherit from `ReferencelessMetric` and have to implement the `compute` function only based on the predictions. The predictions are given in the form of a `Predictions` instance. The corresponding class is implemented in `gem_metrics/texts.py`.
    The `Predictions` object contains the predictions in different formats, for example `list_tokenized` returns a list of tokenized outputs and `list_tokenized_lower` contains them tokenized and lower-cased.
    From there on you should have everything to compute the metric and have the function return either a numeric value or a dictionary, if the metric depends on a hyperparameter and multiple values of it are used, for example.

2. Referenced metric:
For an example, see: [BLEU](https://github.com/GEM-benchmark/GEM-metrics/blob/main/gem_metrics/bleu.py)
    Inherits from `ReferencedMetric` and implements `compute` with an additional `references` parameter. Similar to above, `references` is a `Reference` instance that is also implemented in `gem_metrics/texts.py` and has similar members as the `Predictions` object, just as a list of lists of strings instead of a list of strings.

3. Referenced and sourced metric:
For an example, see: [Sari](https://github.com/GEM-benchmark/GEM-metrics/blob/main/gem_metrics/sari.py)
    Inherits from `SourceAndReferencedMetric` and implements `compute` with an additional `sources` parameter which is a `Source` object defined in `gem_metrics/texts.py`.

After implementing the metric it has to be added to [this dictionary](https://github.com/GEM-benchmark/GEM-metrics/blob/main/gem_metrics/__init__.py#L39) and possibly the default metric list found [here](https://github.com/GEM-benchmark/GEM-metrics/blob/main/gem_metrics/__init__.py#L500), if it is not a heavy metric.

Furthermore, some metrics allow for caching which is switched on and off based on the `support_caching` function that is defined for each metric and `True` by default. Hence, metrics that don't allow caching should override this function to return `False`.

## Automated tests

If you add a new metric, please also add corresponding unit tests under `gem_metrics/tests/test_{metric_name}.py`.
The test data can be found [here](https://github.com/GEM-benchmark/GEM-metrics/tree/main/test_data/unit_tests).

The different metrics inherit from their corresponding baseclasses

1.  `TestReferencedMetric`
2.  `TestSourcedAndReferencedMetric`
3.  `TestReferenceLessMetric`

and `unittest.TestCase`.

Since both classes for referenced metrics already implement basic tests, all you have to do is define the expected results in the `setUp` function after calling `super().setUp()` and setting `self.metric` to an instance of the metric you're about to test.
See [ROGUE](https://github.com/GEM-benchmark/GEM-metrics/blob/main/tests/test_rouge.py) for an example.
Heavy metrics like [BLEURT](https://github.com/GEM-benchmark/GEM-metrics/blob/main/tests/test_bleurt.py) should additionally call the `_initialize()` function of the metric to make sure that the model is properly loaded before tests are executed.

For referenceless metrics, the tests still have to be implemented in the subclasses, see [MSTTR](https://github.com/GEM-benchmark/GEM-metrics/blob/main/tests/test_msttr.py) for an example.


Finally, if you are not testing a heavy metric, make sure to add your test to the list [here](https://github.com/GEM-benchmark/GEM-metrics/blob/main/.github/workflows/main.yml#L30) so that it is run automatically upon the pull request.