import unittest
from tests.inputs import TestData
from gem_metrics.rouge import ROUGE



class BleuTest(unittest.TestCase):

    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        self.rouge_metric = ROUGE()

    def test_rouge_metric(self):
        calculated_metrics = self.rouge_metric.compute(
            references=TestData.references, predictions=TestData.predictions)
        expected_metrics = {'rouge1': {'low': {'precision': 0.4000000000000001, 'recall': 0.2857142857142857, 'fmeasure': 0.3333333333333333}, 'mid': {'precision': 0.673015873015873, 'recall': 0.5777777777777778, 'fmeasure': 0.6203949307397583}, 'high': {'precision': 0.8333333333333334, 'recall': 0.7333333333333333, 'fmeasure': 0.7692307692307692}}, 'rouge2': {'low': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'mid': {'precision': 0.36130536130536123, 'recall': 0.32051282051282054, 'fmeasure': 0.3395061728395062}, 'high': {'precision': 0.5454545454545454, 'recall': 0.5, 'fmeasure': 0.5185185185185186}}, 'rougeL': {'low': {'precision': 0.20000000000000004, 'recall': 0.14285714285714285, 'fmeasure': 0.16666666666666666}, 'mid': {'precision': 0.5785714285714286, 'recall': 0.5063492063492063, 'fmeasure': 0.5391983495431771}, 'high': {'precision': 0.7857142857142857, 'recall': 0.7333333333333333, 'fmeasure': 0.7586206896551723}}, 'rougeLsum': {'low': {'precision': 0.20000000000000004, 'recall': 0.14285714285714285, 'fmeasure': 0.16666666666666666}, 'mid': {'precision': 0.5785714285714286, 'recall': 0.5063492063492063, 'fmeasure': 0.5391983495431771}, 'high': {'precision': 0.7857142857142857, 'recall': 0.7333333333333333, 'fmeasure': 0.7586206896551723}}}
        self.assertDictEqual(calculated_metrics, expected_metrics)
        
    def test_rouge_metric_identical_pred_ref(self):
        """Tests for identical predictions and references (ROUGE = 1.0)
        """

        calculated_metrics = self.rouge_metric.compute(
            references=TestData.references, predictions=TestData.identical_predictions)
        expected_metrics = {'rouge1': {'low': {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}, 'mid': {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}, 'high': {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}}, 'rouge2': {'low': {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}, 'mid': {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}, 'high': {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}}, 'rougeL': {'low': {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}, 'mid': {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}, 'high': {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}}, 'rougeLsum': {'low': {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}, 'mid': {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}, 'high': {'precision': 1.0, 'recall': 1.0, 'fmeasure': 1.0}}}
        self.assertDictEqual(calculated_metrics, expected_metrics)

    def test_rouge_metric_mismatched_pred_ref(self):
        """Tests for completely dissimilar predictions and references (ROUGE = 0.0)
        """
        calculated_metrics = self.rouge_metric.compute(references=TestData.references,
                                                      predictions=TestData.reversed_predictions)
        expected_metrics = {'rouge1': {'low': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'mid': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'high': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}}, 'rouge2': {'low': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'mid': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'high': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}}, 'rougeL': {'low': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'mid': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'high': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}}, 'rougeLsum': {'low': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'mid': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'high': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}}}
        self.assertDictEqual(calculated_metrics, expected_metrics)

    def test_bleu_metric_mismatched_empty_tgt(self):
        """Tests for empty target (ROUGE = 0.0)
        """
        tgt_empty = [
            "",
            "",
            ""
        ]
        calculated_metrics = self.rouge_metric.compute(
            references=TestData.references, predictions=TestData.empty_predictions)
        expected_metrics = {'rouge1': {'low': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'mid': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'high': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}}, 'rouge2': {'low': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'mid': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'high': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}}, 'rougeL': {'low': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'mid': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'high': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}}, 'rougeLsum': {'low': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'mid': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}, 'high': {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0}}}
        self.assertDictEqual(calculated_metrics, expected_metrics)


if __name__ == '__main__':
    unittest.main()
