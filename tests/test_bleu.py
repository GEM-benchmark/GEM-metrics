import unittest
from tests.inputs import TestData
from gem_metrics.bleu import BLEU
from gem_metrics.texts import Predictions, References


class BleuTest(unittest.TestCase):

    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        self.bleu_metric = BLEU()

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
        self.assertAlmostEqual(calculated_metrics["bleu"], 0.53, places=1)

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
