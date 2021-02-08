import json
import os
import unittest
from gem_metrics.ngrams import NGramStats
from gem_metrics.texts import Predictions

# TODO: add multilingual tests

class NGramTest(unittest.TestCase):

    test_rust_tokenizer = True

    def setUp(self):
        super().setUp()

        self.ngram_metric = NGramStats()

    def test_ngram_metric_basic(self):
        """Tests for the base case.
        """
        text = [
            "Alimentum is not family-friendly, and is near the Burger King in the city centre.",
            "There is a place in the city centre, Alimentum, that is not family-friendly.",
            "There is a house in New Orleans."
        ]
        test_predictions = Predictions({"values": text, "language": "en"})

        expected_metrics = {'total_length': 40, 'mean_pred_length': 13.333333333333334, 'std_pred_length': 3.7712361663282534, 'median_pred_length': 16.0, 'min_pred_length': 8, 'max_pred_length': 16, 'distinct-1': 0.525, 'vocab_size-1': 21, 'unique-1': 9, 'entropy-1': 4.206198332810095, 'distinct-2': 0.8108108108108109, 'vocab_size-2': 30, 'unique-2': 23, 'entropy-2': 4.831074987250575, 'cond_entropy-2': 0.5868307567125932, 'distinct-3': 0.8823529411764706, 'vocab_size-3': 30, 'unique-3': 26, 'entropy-3': 4.852168723603279, 'cond_entropy-3': 0.05448006385668387, 'total_length-nopunct': 34, 'mean_pred_length-nopunct': 11.333333333333334, 'std_pred_length-nopunct': 3.0912061651652345, 'median_pred_length-nopunct': 13.0, 'min_pred_length-nopunct': 7, 'max_pred_length-nopunct': 14, 'distinct-1-nopunct': 0.5588235294117647, 'vocab_size-1-nopunct': 19, 'unique-1-nopunct': 9, 'entropy-1-nopunct': 4.054538856580818, 'distinct-2-nopunct': 0.7741935483870968, 'vocab_size-2-nopunct': 24, 'unique-2-nopunct': 17, 'entropy-2-nopunct': 4.50258340716107, 'cond_entropy-2-nopunct': 0.48348880716117293, 'distinct-3-nopunct': 0.8571428571428571, 'vocab_size-3-nopunct': 24, 'unique-3-nopunct': 20, 'entropy-3-nopunct': 4.52164063634332, 'cond_entropy-3-nopunct': -0.003984245472128352}
        
        calculated_metrics = self.ngram_metric.compute(test_predictions)
        self.assertDictEqual(expected_metrics, calculated_metrics)

    

    def test_ngram_metric_empty(self):
        """Tests with empty inputs
        """
        text = [
            "",
            ""
        ]
        calculated_metrics = self.ngram_metric.compute(Predictions({"values": text, "language": "en"}))        
        
        expected_metrics = {'total_length': 0, 'mean_pred_length': 0.0, 'std_pred_length': 0.0, 'median_pred_length': 0.0, 'min_pred_length': 0, 'max_pred_length': 0, 'distinct-1': 0, 'vocab_size-1': 0, 'unique-1': 0, 'entropy-1': 0, 'distinct-2': 0, 'vocab_size-2': 0, 'unique-2': 0, 'entropy-2': 0, 'distinct-3': 0, 'vocab_size-3': 0, 'unique-3': 0, 'entropy-3': 0, 'total_length-nopunct': 0, 'mean_pred_length-nopunct': 0.0, 'std_pred_length-nopunct': 0.0, 'median_pred_length-nopunct': 0.0, 'min_pred_length-nopunct': 0, 'max_pred_length-nopunct': 0, 'distinct-1-nopunct': 0, 'vocab_size-1-nopunct': 0, 'unique-1-nopunct': 0, 'entropy-1-nopunct': 0, 'distinct-2-nopunct': 0, 'vocab_size-2-nopunct': 0, 'unique-2-nopunct': 0, 'entropy-2-nopunct': 0, 'distinct-3-nopunct': 0, 'vocab_size-3-nopunct': 0, 'unique-3-nopunct': 0, 'entropy-3-nopunct': 0}

        self.assertDictEqual(expected_metrics, calculated_metrics)


if __name__ == '__main__':
    unittest.main()
