"""Test class for referenced tests
"""
import unittest
from tests.inputs import TestData
from gem_metrics.bleu import BLEU


class ReferencedMetricTest(unittest.TestCase):

    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        self.metric = BLEU()
        

    def test_metric(self):
        """Tests for identical predictions and references (BLEU = 100)
        """

        calculated_metrics = self.metric.compute(
            references=TestData.references, predictions=TestData.predictions)
        self.assertDictEqual(calculated_metrics, self.expected_result_basic, places=2)

    def test_metric_identical_pred_ref(self):
        """Tests for identical predictions and references (BLEU = 100)
        """

        calculated_metrics = self.metric.compute(
            references=TestData.references, predictions=TestData.identical_predictions)
        self.assertDictEqual(calculated_metrics, self.expected_result_identical_pred_ref)

    def test_metric_mismatched_pred_ref(self):
        """Tests for completely dissimilar predictions and references (BLEU = 0.)
        """
        calculated_metrics = self.metric.compute(references=TestData.references,
                                                      predictions=TestData.reversed_predictions)
        #  TODO: check why BLEU is not exactly 0 (smoothing?)
        self.assertDictEqual(calculated_metrics, self.expected_result_mismatched_pred_ref, places=2)

    def test_metric_mismatched_empty_tgt(self):
        """Tests for empty target
        """
        tgt_empty = [
            "",
            "",
            ""
        ]
        calculated_metrics = self.metric.compute(
            references=TestData.references, predictions=TestData.empty_predictions)
        self.assertDictEqual(calculated_metrics, self.expected_result_empty_pred_ref)


if __name__ == '__main__':
    unittest.main()
