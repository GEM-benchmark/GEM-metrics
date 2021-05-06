import unittest
from gem_metrics.ngrams import NGramStats
from gem_metrics.texts import Predictions
from tests.test_referenceless import TestReferenceLessMetric
from tests.inputs import TestData
from tests.utils import assertDeepAlmostEqual

# TODO: add multilingual tests


class TestNGram(TestReferenceLessMetric, unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.ngram_metric = NGramStats()

    def test_ngram_metric_basic(self):
        """Tests for the base case.
        """
        expected_metrics = {'total_length': 33, 'mean_pred_length': 11.0, 'std_pred_length': 3.559026084010437, 'median_pred_length': 13.0, 'min_pred_length': 6, 'max_pred_length': 14, 'distinct-1': 0.6363636363636364, 'vocab_size-1': 21, 'unique-1': 12, 'entropy-1': 4.233639952626228, 'distinct-2': 0.8333333333333334, 'vocab_size-2': 25, 'unique-2': 20, 'entropy-2': 4.573557262275186, 'cond_entropy-2': 0.22099272632218087, 'distinct-3': 0.8888888888888888, 'vocab_size-3': 24, 'unique-3': 21, 'entropy-3': 4.532665279941249, 'cond_entropy-3': -0.04089198233393895, 'total_length-nopunct': 29, 'mean_pred_length-nopunct': 9.666666666666666,
                            'std_pred_length-nopunct': 3.39934634239519, 'median_pred_length-nopunct': 11.0, 'min_pred_length-nopunct': 5, 'max_pred_length-nopunct': 13, 'distinct-1-nopunct': 0.6896551724137931, 'vocab_size-1-nopunct': 20, 'unique-1-nopunct': 12, 'entropy-1-nopunct': 4.21126073643228, 'distinct-2-nopunct': 0.8461538461538461, 'vocab_size-2-nopunct': 22, 'unique-2-nopunct': 18, 'entropy-2-nopunct': 4.392747410448783, 'cond_entropy-2-nopunct': 0.1791851654044227, 'distinct-3-nopunct': 0.9130434782608695, 'vocab_size-3-nopunct': 21, 'unique-3-nopunct': 19, 'entropy-3-nopunct': 4.349648912578752, 'cond_entropy-3-nopunct': -0.04644297947538351}

        calculated_metrics = self.ngram_metric.compute(TestData.predictions)
        assertDeepAlmostEqual(self, expected_metrics, calculated_metrics)

    def test_ngram_metric_empty(self):
        """Tests with empty inputs
        """
        text = [
            "",
            ""
        ]

        expected_metrics = {'total_length': 0, 'mean_pred_length': 0.0, 'std_pred_length': 0.0, 'median_pred_length': 0.0, 'min_pred_length': 0, 'max_pred_length': 0, 'distinct-1': 0, 'vocab_size-1': 0, 'unique-1': 0, 'entropy-1': 0, 'distinct-2': 0, 'vocab_size-2': 0, 'unique-2': 0, 'entropy-2': 0, 'distinct-3': 0, 'vocab_size-3': 0, 'unique-3': 0, 'entropy-3': 0, 'total_length-nopunct': 0, 'mean_pred_length-nopunct': 0.0,
                            'std_pred_length-nopunct': 0.0, 'median_pred_length-nopunct': 0.0, 'min_pred_length-nopunct': 0, 'max_pred_length-nopunct': 0, 'distinct-1-nopunct': 0, 'vocab_size-1-nopunct': 0, 'unique-1-nopunct': 0, 'entropy-1-nopunct': 0, 'distinct-2-nopunct': 0, 'vocab_size-2-nopunct': 0, 'unique-2-nopunct': 0, 'entropy-2-nopunct': 0, 'distinct-3-nopunct': 0, 'vocab_size-3-nopunct': 0, 'unique-3-nopunct': 0, 'entropy-3-nopunct': 0}

        calculated_metrics = self.ngram_metric.compute(
            Predictions({"values": text, "language": "en"}))

        assertDeepAlmostEqual(self, expected_metrics, calculated_metrics)

    def test_ngram_metric_mixed(self):
        """Tests with empty inputs
        """
        text = [
            "",
            "token number one token number two token number three",
            ""
        ]

        expected_metrics = {'total_length': 9, 'mean_pred_length': 3.0, 'std_pred_length': 4.242640687119285, 'median_pred_length': 0.0, 'min_pred_length': 0, 'max_pred_length': 9, 'distinct-1': 0.5555555555555556, 'vocab_size-1': 5, 'unique-1': 3, 'entropy-1': 2.113283334294875, 'distinct-2': 0.75, 'vocab_size-2': 6, 'unique-2': 5, 'entropy-2': 2.4056390622295662, 'cond_entropy-2': 0.42443593632812127, 'distinct-3': 1.0, 'vocab_size-3': 7, 'unique-3': 7, 'entropy-3': 2.807354922057604, 'cond_entropy-3': 0.486624565223814, 'total_length-nopunct': 9, 'mean_pred_length-nopunct': 3.0,
                            'std_pred_length-nopunct': 4.242640687119285, 'median_pred_length-nopunct': 0.0, 'min_pred_length-nopunct': 0, 'max_pred_length-nopunct': 9, 'distinct-1-nopunct': 0.5555555555555556, 'vocab_size-1-nopunct': 5, 'unique-1-nopunct': 3, 'entropy-1-nopunct': 2.113283334294875, 'distinct-2-nopunct': 0.75, 'vocab_size-2-nopunct': 6, 'unique-2-nopunct': 5, 'entropy-2-nopunct': 2.4056390622295662, 'cond_entropy-2-nopunct': 0.42443593632812127, 'distinct-3-nopunct': 1.0, 'vocab_size-3-nopunct': 7, 'unique-3-nopunct': 7, 'entropy-3-nopunct': 2.807354922057604, 'cond_entropy-3-nopunct': 0.486624565223814}

        calculated_metrics = self.ngram_metric.compute(
            Predictions({"values": text, "language": "en"}))
        assertDeepAlmostEqual(self, expected_metrics, calculated_metrics)

    def test_ngram_metric_repeated(self):
        """Tests with empty inputs
        """
        text = [
            "token token token token token token token"
        ]

        expected_metrics = {'total_length': 7, 'mean_pred_length': 7.0, 'std_pred_length': 0.0, 'median_pred_length': 7.0, 'min_pred_length': 7, 'max_pred_length': 7, 'distinct-1': 0.14285714285714285, 'vocab_size-1': 1, 'unique-1': 0, 'entropy-1': -0.0, 'distinct-2': 0.16666666666666666, 'vocab_size-2': 1, 'unique-2': 0, 'entropy-2': -0.0, 'cond_entropy-2': -0.0, 'distinct-3': 0.2, 'vocab_size-3': 1, 'unique-3': 0, 'entropy-3': -0.0, 'cond_entropy-3': -0.0, 'total_length-nopunct': 7, 'mean_pred_length-nopunct': 7.0,
                            'std_pred_length-nopunct': 0.0, 'median_pred_length-nopunct': 7.0, 'min_pred_length-nopunct': 7, 'max_pred_length-nopunct': 7, 'distinct-1-nopunct': 0.14285714285714285, 'vocab_size-1-nopunct': 1, 'unique-1-nopunct': 0, 'entropy-1-nopunct': -0.0, 'distinct-2-nopunct': 0.16666666666666666, 'vocab_size-2-nopunct': 1, 'unique-2-nopunct': 0, 'entropy-2-nopunct': -0.0, 'cond_entropy-2-nopunct': -0.0, 'distinct-3-nopunct': 0.2, 'vocab_size-3-nopunct': 1, 'unique-3-nopunct': 0, 'entropy-3-nopunct': -0.0, 'cond_entropy-3-nopunct': -0.0}

        calculated_metrics = self.ngram_metric.compute(
            Predictions({"values": text, "language": "en"}))
        assertDeepAlmostEqual(self, expected_metrics, calculated_metrics)


if __name__ == '__main__':
    unittest.main()
