import unittest
import gem_metrics.nubia
from tests.test_referenced import TestReferencedMetric


class TestNubia(TestReferencedMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.nubia.NUBIA()
        self.true_results_basic = {'nubia': 0.767}
        self.true_results_identical_pred_ref = {'nubia': 1.0}
        self.true_results_mismatched_pred_ref = {'nubia': 0.047}
        self.true_results_empty_pred = {'nubia': 0}


if __name__ == '__main__':
    unittest.main()
