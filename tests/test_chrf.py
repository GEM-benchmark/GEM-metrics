import unittest
import gem_metrics.chrf
from tests.test_referenced import TestReferencedMetric


class TestCHRF(TestReferencedMetric, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.chrf.CHRF()
        self.true_results_basic = {
            "chrf": 63.61553,
            "chrf+": 63.40982,
            "chrf++": 60.0482,
        }
        self.true_results_identical_pred_ref = {
            "chrf": 100.0,
            "chrf+": 100.0,
            "chrf++": 100.0,
        }
        self.true_results_mismatched_pred_ref = {
            "chrf": 19.31088,
            "chrf+": 16.55218,
            "chrf++": 14.48316,
        }
        self.true_results_empty_pred = {"chrf": 0.0, "chrf+": 0.0, "chrf++": 0.0}


if __name__ == "__main__":
    unittest.main()
