import unittest
import sys
import gem_metrics.bleurt
from tests.test_referenced import TestReferencedMetric

sys.argv = sys.argv[:1]  # ignore unittest flags

class TestBluert(TestReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.bleurt.BLEURT()
        self.true_results_basic = {'bleurt': -0.018821040789286297}
        self.true_results_identical_pred_ref = {'bleurt': 0.941490113735199}
        self.true_results_mismatched_pred_ref = {'bleurt': -1.2387781937917073}
        self.true_results_empty_pred = {'bleurt': -2.01493509610494}

if __name__ == '__main__':
    unittest.main()
