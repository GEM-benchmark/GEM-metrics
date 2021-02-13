import unittest
import gem_metrics
from tests.test_referenced import TestReferencedMetric




class TestBleu(TestReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.bleu.BLEU()
        self.expected_result_basic = {'bleu': 32.56}
        self.expected_result_identical_pred_ref = {'bleu': 100}
        #  TODO: check why BLEU is not exactly 0 (smoothing?)
        self.expected_result_mismatched_pred_ref = {'bleu': 0.505}
        self.expected_result_empty_pred_ref = {'bleu': 0}


if __name__ == '__main__':
    unittest.main()
