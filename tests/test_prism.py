import unittest
import gem_metrics.prism
from tests.test_referenced import TestReferencedMetric


class TestPrism(TestReferencedMetric, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.prism.Prism(device=-1)
        self.true_results_basic = {"prism": -1.7404}
        self.true_results_identical_pred_ref = {"prism": -0.229}
        self.true_results_mismatched_pred_ref = {"prism": -6.62654}
        self.true_results_empty_pred = {"prism": -9.90867}


if __name__ == "__main__":
    unittest.main()
