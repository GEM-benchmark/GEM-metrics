import unittest
import gem_metrics.ter
from tests.test_referenced import TestReferencedMetric


class TestTer(TestReferencedMetric, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.ter.TER()
        self.true_results_basic = {"ter": 52.94118}
        self.true_results_identical_pred_ref = {"ter": 0.0}
        self.true_results_mismatched_pred_ref = {"ter": 102.94117}
        self.true_results_empty_pred = {"ter": 100.0}


if __name__ == "__main__":
    unittest.main()