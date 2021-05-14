import unittest
import gem_metrics.meteor
from tests.test_referenced import TestReferencedMetric

class TestMeteor(TestReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.Meteor()
        self.true_results_basic = {'meteor': 0.42}
        self.true_results_identical_pred_ref = {'meteor': 1.}
        self.true_results_mismatched_pred_ref = {'meteor': 0.}
        self.true_results_empty_pred = {'meteor': 0.}

if __name__ == '__main__':
    unittest.main()
