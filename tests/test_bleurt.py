import unittest
import gem_metrics
from tests.test_referenced import TestReferencedMetric

class TestBluert(TestReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.bleurt.BLEURT()
        self.expected_result_basic = {'bleurt': -0.018821040789286297}
        self.expected_result_identical_pred_ref = {'bleurt': 0.941490113735199}
        self.expected_result_mismatched_pred_ref = {'bleurt': -1.2387781937917073}
        self.expected_result_empty_pred_ref = {'bleurt': -2.01493509610494}

if __name__ == '__main__':
    unittest.main()
