import unittest
import gem_metrics
from tests.test_referenced import ReferencedMetricTest

class TestLocalRecall(ReferencedMetricTest, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.local_recall.LocalRecall()
        self.expected_result_basic = {'local_recall': {1: 0.6451612903225806}}
        self.expected_result_identical_pred_ref = {'local_recall': {1: 1.0}}
        self.expected_result_mismatched_pred_ref = {'local_recall': {1: 0.0}}
        self.expected_result_empty_pred_ref = {'local_recall': {1: 0.0}}

if __name__ == '__main__':
    unittest.main()
