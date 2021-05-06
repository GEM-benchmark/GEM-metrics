import unittest
import gem_metrics.bleu
from tests.test_referenced import TestReferencedMetric




class TestBleu(TestReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.bleu.BLEU()
        self.true_results_basic = {'bleu': 32.56}
        self.true_results_identical_pred_ref = {'bleu': 100}
        #  TODO: check why BLEU is not exactly 0 (smoothing?)
        self.true_results_mismatched_pred_ref = {'bleu': 0.505}
        self.true_results_empty_pred = {'bleu': 0}


if __name__ == '__main__':
    unittest.main()
