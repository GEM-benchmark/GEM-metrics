import unittest
import gem_metrics.questeval
from tests.test_sourced_and_referenced import TestSourcedAndReferencedMetric


class TestQuestEval(TestSourcedAndReferencedMetric, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.questeval.QuestEval()
        self.true_results_basic = {"questeval": 0.7226280048489571}
        self.true_results_identical_pred_ref = {"questeval": 0.877193151248826}
        self.true_results_mismatched_pred_ref = {"questeval": 0.22809108622648097}
        self.true_results_empty_pred = {"questeval": 0.0}


if __name__ == "__main__":
    unittest.main()
