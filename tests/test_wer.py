import unittest
import gem_metrics.wer
from tests.test_referenced import TestReferencedMetric


class TestWer(TestReferencedMetric, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.wer.WER()
        self.true_results_basic = {"wer": 58.05861}
        self.true_results_identical_pred_ref = {"wer": 0.0}
        self.true_results_mismatched_pred_ref = {"wer": 102.38095}
        self.true_results_empty_pred = {"wer": 100.0}


if __name__ == "__main__":
    unittest.main()