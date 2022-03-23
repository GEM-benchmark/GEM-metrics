import unittest
import gem_metrics.moverscore
from tests.test_referenced import TestReferencedMetric


class TestMoverScore(TestReferencedMetric, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.moverscore.MoverScore()
        self.true_results_basic = {"moverscore": 0.65521}
        self.true_results_identical_pred_ref = {"moverscore": 0.99975}
        self.true_results_mismatched_pred_ref = {"moverscore": 0.49}
        self.true_results_empty_pred = {"moverscore": 0.44899}


if __name__ == "__main__":
    unittest.main()

