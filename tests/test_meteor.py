import unittest
import gem_metrics
from tests.test_referenced import TestReferencedMetric

class TestMeteor(TestReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.Meteor()
        self.expected_result_basic = {'meteor': 0.42}
        self.expected_result_identical_pred_ref = {'meteor': 1.}
        self.expected_result_mismatched_pred_ref = {'meteor': 0.}
        self.expected_result_empty_pred_ref = {'meteor': 0.}

if __name__ == '__main__':
    unittest.main()
