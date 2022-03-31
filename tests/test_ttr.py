import unittest
import math
from gem_metrics.ttr import TTR
from gem_metrics.texts import Predictions
from tests.test_referenceless import TestReferenceLessMetric
from tests.inputs import TestData
from tests.utils import assertDeepAlmostEqual


class TestTTR(TestReferenceLessMetric, unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.metric = TTR()

    def test_ttr_metric_basic(self):
        """Tests for the base case."""

        expected_metrics = {
            "ttr": 0.72414
        }

        calculated_metrics = self.metric.compute({}, TestData.predictions)
        assertDeepAlmostEqual(self, expected_metrics, calculated_metrics)

    def test_ttr_metric_empty(self):
        """Tests with list of empty sentences. """
        text = ["", ""]

        calculated_metrics = self.metric.compute(
            {},
            Predictions({"values": text})
        )

        self.assertTrue(math.isnan(calculated_metrics["ttr"]))

    def test_ttr_disjoint_tokens(self):
        """Tests for TTR with disjoint tokens. The TTR should be 1.
        """
        text = [
            "one two three four five six seven eight nine ten",
            "eleven twelve thirteen fourteen fifteen sixteen",
        ]

        metric = TTR()
        calculated_metrics = metric.compute(
            {},
            Predictions({"values": text})
        )
        self.assertAlmostEqual(calculated_metrics[f"ttr"], 1)

    def test_ttr_identical_tokens(self):
        """Tests for TTR with identical tokens.
        As the tokens are identical, the TTR be 1 / total_tokens (there is only one _type_ of token)
        """
        text = [
            "token token token token token token token token token token token token token",
            "token token token token token token token token token token token token token",
        ]

        metric = TTR()
        calculated_metrics = metric.compute(
            {},
            Predictions({"values": text})
        )
        self.assertAlmostEqual(
            calculated_metrics[f"ttr"], round(1/sum(len(s.split()) for s in text), 5)
        )

if __name__ == "__main__":
    unittest.main()
