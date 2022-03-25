import unittest
import gem_metrics.cider
from tests.test_referenced import TestReferencedMetric


class TestCider(TestReferencedMetric, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.cider.CIDER()
        self.true_results_basic = {"CIDEr": 1.89}
        self.true_results_identical_pred_ref = {"CIDEr": 10.0}
        self.true_results_mismatched_pred_ref = {"CIDEr": 0.0}
        self.true_results_empty_pred = {"CIDEr": 0.0}


if __name__ == "__main__":
    unittest.main()