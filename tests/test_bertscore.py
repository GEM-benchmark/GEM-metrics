import unittest
import gem_metrics
from tests.test_referenced import TestReferencedMetric

class TestRouge(TestReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.bertscore.BERTScore()
        self.expected_result_basic = {'bertscore': 0.42}
        self.expected_result_identical_pred_ref = {'bertscore': 1.}
        self.expected_result_mismatched_pred_ref = {'bertscore': 0.}
        self.expected_result_empty_pred_ref = {'bertscore': 0.}

if __name__ == '__main__':
    unittest.main()
