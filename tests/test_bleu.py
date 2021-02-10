import unittest

from numpy.lib.function_base import place
from tests.inputs import TestData
from tests.test_referenced import ReferencedMetricTest
from gem_metrics.bleu import BLEU



class BleuTest2(ReferencedMetricTest):

    def setUp(self):
        super().setUp()
        self.expected_result_basic = {'bleu': 32.56}
        self.expected_result_identical_pred_ref = {'bleu': 100}
        self.expected_result_mismatched_pred_ref = {'bleu': 0.505}
        self.expected_result_empty_pred_ref = {'bleu': 0}
        

class BleuTest():

    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        self.bleu_metric = BLEU()

    def test_bleu_metric(self):
        """Tests for identical predictions and references (BLEU = 100)
        """

        calculated_metrics = self.bleu_metric.compute(
            references=TestData.references, predictions=TestData.predictions)
        self.assertAlmostEqual(calculated_metrics["bleu"], 32.56, places=2)

    def test_bleu_metric_identical_pred_ref(self):
        """Tests for identical predictions and references (BLEU = 100)
        """

        calculated_metrics = self.bleu_metric.compute(
            references=TestData.references, predictions=TestData.identical_predictions)
        self.assertAlmostEqual(calculated_metrics["bleu"], 100.0)

    def test_bleu_metric_mismatched_pred_ref(self):
        """Tests for completely dissimilar predictions and references (BLEU = 0.)
        """
        calculated_metrics = self.bleu_metric.compute(references=TestData.references,
                                                      predictions=TestData.reversed_predictions)
        #  TODO: check why BLEU is not exactly 0 (smoothing?)
        self.assertAlmostEqual(calculated_metrics["bleu"], 0.505, places=2)

    def test_bleu_metric_mismatched_empty_tgt(self):
        """Tests for empty target
        """
        tgt_empty = [
            "",
            "",
            ""
        ]
        calculated_metrics = self.bleu_metric.compute(
            references=TestData.references, predictions=TestData.empty_predictions)
        self.assertAlmostEqual(calculated_metrics["bleu"], 0)


if __name__ == '__main__':
    unittest.main()
