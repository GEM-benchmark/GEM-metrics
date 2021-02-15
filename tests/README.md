Tests for GEM-metrics [WIP]
====

Running all the tests
-------------

From the base directory:
```sh
python -m unittest -v
```


Code Organization
-------------
- There are three base test classes, one for each metric type:

    1.  `TestReferencedMetric`
    2.  `TestSourcedAndReferencedMetric`
    3.  `TestReferenceLessMetric`

- Each class (except `TestReferenceLessMetric`) defines some basic test cases that an instance metric should satisfy.


Adding tests for a new metric
-------------

- To add tests for a new metric, create an instance of the right class (i.e., class corresponding to the metric type) and then _fill in_ the expected results for the basic test cases in `setUp()`.
- Example from `test_meteor.py`:
```py
    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.Meteor()
        self.true_results_basic = {'meteor': 0.42}
        self.true_results_identical_pred_ref = {'meteor': 1.}
        self.true_results_mismatched_pred_ref = {'meteor': 0.}
        self.true_results_empty_pred = {'meteor': 0.}
```
- Please see `tests/inputs.py` for examples of the test data.
- You can override any of the functions in the base test classes (`TestReferenceLessMetric`) and add new, metric specific cases.
- `TestReferenceLessMetric` currently contains no standardized test cases, but we still recommend using it for consistency.