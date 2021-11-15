import unittest
import math
from gem_metrics.msttr import MSTTR
from gem_metrics.texts import Predictions
from tests.test_referenceless import TestReferenceLessMetric
from tests.inputs import TestData
from tests.utils import assertDeepAlmostEqual

# TODO: add multilingual tests


class TestMSTTR(TestReferenceLessMetric, unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.metric = MSTTR(window_size=4)

    def test_msttr_metric_basic(self):
        """Tests for the base case."""
        # for window size 4, the test data has only segments with unique tokens
        self.metric.window_size = 12
        expected_metrics = {
            "msttr-12": 0.875,
            "msttr-12_nopunct": 0.91667,
        }

        calculated_metrics = self.metric.compute({}, TestData.predictions)

        assertDeepAlmostEqual(self, expected_metrics, calculated_metrics)

    def test_msttr_metric_empty(self):
        """Tests with empty inputs"""
        text = ["", ""]

        calculated_metrics = self.metric.compute(
            {},
            Predictions({"values": text, "language": "en"})
        )

        self.assertTrue(math.isnan(calculated_metrics["msttr-4"]))
        self.assertTrue(math.isnan(calculated_metrics["msttr-4_nopunct"]))

    def test_msttr_disjoint_tokens(self):
        """Tests for MSTTR with disjoint tokens and by varying the window size.
        The MSTTR with window size w should be 1 (every window has unique tokens).
        """
        text = [
            "one two three four five six seven eight nine ten",
            "eleven twelve thirteen fourteen fifteen sixteen",
        ]
        for window_size in range(1, 11):
            metric = MSTTR(window_size=window_size)
            calculated_metrics = metric.compute(
                {},
                Predictions({"values": text, "language": "en"})
            )
            self.assertAlmostEquals(calculated_metrics[f"msttr-{window_size}"], 1)

    def test_msttr_identical_tokens(self):
        """Tests for MSTTR with identical tokens and by varying the window size.
        As the tokens are identical, the MSTTR with window size w should be 1 / w (there is only one _type_ of token)
        """
        text = [
            "token token token token token token token token token token token token token",
            "token token token token token token token token token token token token token",
        ]
        for window_size in range(1, 11):
            metric = MSTTR(window_size=window_size)
            calculated_metrics = metric.compute(
                {},
                Predictions({"values": text, "language": "en"})
            )
            self.assertAlmostEqual(
                calculated_metrics[f"msttr-{window_size}"], round(1 / window_size, 5)
            )


if __name__ == "__main__":
    unittest.main()
