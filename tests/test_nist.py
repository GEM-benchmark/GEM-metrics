import unittest
import gem_metrics.nist
from tests.test_referenced import TestReferencedMetric


class TestNist(TestReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.nist.NIST()
        self.true_results_basic = {'nist': 2.949}
        self.true_results_identical_pred_ref = {'nist': 5.346}
        self.true_results_mismatched_pred_ref = {'nist': 0}
        self.true_results_empty_pred = {'nist': 0}


if __name__ == '__main__':
    unittest.main()
