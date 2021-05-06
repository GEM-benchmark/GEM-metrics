import unittest
import gem_metrics.bertscore
from tests.test_referenced import TestReferencedMetric

class TestBertScore(TestReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.bertscore.BERTScore()
        self.metrics_keys_to_ignore = {"bertscore": ["hashcode"]}
        self.true_results_basic = {'bertscore': {'precision': 0.9004354476928711, 'recall': 0.8788148760795593, 'f1': 0.8894707560539246}}
        self.true_results_identical_pred_ref = {'bertscore': {'precision': 0.9999999006589254, 'recall': 0.9999999006589254, 'f1': 0.9999999006589254}}
        self.true_results_mismatched_pred_ref = {'bertscore': {'precision': 0.6620853543281555, 'recall': 0.6638288497924805, 'f1': 0.6626873016357422}}
        self.true_results_empty_pred = {'bertscore': {'precision': 0.0, 'recall': 0.5345260302225748, 'f1': 0.0}}


if __name__ == '__main__':
    unittest.main()
