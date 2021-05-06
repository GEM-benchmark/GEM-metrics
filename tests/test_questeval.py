import unittest
import sys
import gem_metrics.questeval
from tests.test_sourced_and_referenced import TestSourcedAndReferencedMetric


class TestQuestEval(TestSourcedAndReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.questeval.QuestEval()
        self.true_results_basic = {'questeval': {'precision': 0.8430564959433321, 'recall': 0.7523951736176925, 'f1': 0.7977258347805124}}
        self.true_results_identical_pred_ref = {'questeval': {'precision': 0.8886355747320557, 'recall': 0.8886355747320557, 'f1': 0.8886355747320557}}
        self.true_results_mismatched_pred_ref = {'questeval': {'precision': 0.23941414190663232, 'recall': 0.2270226373716637, 'f1': 0.23321838963914798}}
        self.true_results_empty_pred = {'questeval': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}}


if __name__ == '__main__':
    unittest.main()
