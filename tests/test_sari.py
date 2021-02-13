from tests.test_sourced_and_referenced import TestSourcedAndReferencedMetric
import unittest
import gem_metrics
from tests.test_sourced_and_referenced import TestSourcedAndReferencedMetric




class TestSari(TestSourcedAndReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.sari.SARI()
        self.expected_result_basic = {'sari': 49.37}
        self.expected_result_identical_pred_ref = {'sari': 99.07}
        #  TODO: check why BLEU is not exactly 0 (smoothing?)
        self.expected_result_mismatched_pred_ref = {'sari': 29.33}
        self.expected_result_empty_pred_ref = {'sari': 29.33}


if __name__ == '__main__':
    unittest.main()
