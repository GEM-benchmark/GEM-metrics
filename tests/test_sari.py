from tests.test_sourced_and_referenced import TestSourcedAndReferencedMetric
import unittest
import gem_metrics.sari
from tests.test_sourced_and_referenced import TestSourcedAndReferencedMetric




class TestSari(TestSourcedAndReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.sari.SARI()
        self.true_results_basic = {'sari': 49.37}
        self.true_results_identical_pred_ref = {'sari': 99.07}
        self.true_results_mismatched_pred_ref = {'sari': 29.33}
        self.true_results_empty_pred = {'sari': 29.33}


if __name__ == '__main__':
    unittest.main()
