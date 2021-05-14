import unittest
import gem_metrics.local_recall
from tests.test_referenced import TestReferencedMetric

class TestLocalRecall(TestReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.local_recall.LocalRecall()
        self.true_results_basic = {'local_recall': {1: 0.6451612903225806}}
        self.true_results_identical_pred_ref = {'local_recall': {1: 1.0}}
        self.true_results_mismatched_pred_ref = {'local_recall': {1: 0.0}}
        self.true_results_empty_pred = {'local_recall': {1: 0.0}}

if __name__ == '__main__':
    unittest.main()
