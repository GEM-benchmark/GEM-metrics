import unittest
from gem_metrics.yules_i import Yules_I
from gem_metrics.texts import Predictions
from tests.test_referenceless import TestReferenceLessMetric
from tests.inputs import TestData
from tests.utils import assertDeepAlmostEqual


class TestYULES_I(TestReferenceLessMetric, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.metric = Yules_I()

    def test_yules_i_metric_basic(self):
        """Tests for the base case."""
        expected_metrics = {"yules_i": 16.962}

        calculated_metrics = self.metric.compute({}, TestData.predictions)
        assertDeepAlmostEqual(self, expected_metrics, calculated_metrics)

    def test_yules_i_metric_empty(self):
        """Tests with empty inputs"""
        text = ["", ""]

        calculated_metrics = self.metric.compute({}, Predictions({"values": text}))

        self.assertAlmostEqual(calculated_metrics[f"yules_i"], 0)

    def test_yules_i_disjoint_tokens(self):
        """Tests for Yules_I with disjoint tokens."""
        text = [
            "one two three four five six seven eight nine ten",
            "eleven twelve thirteen fourteen fifteen sixteen",
        ]
        metric = Yules_I()
        calculated_metrics = metric.compute({}, Predictions({"values": text}))
        self.assertAlmostEqual(calculated_metrics[f"yules_i"], 0.0)

    def test_yules_i_mixed_tokens(self):
        """Tests for Yules_I with disjoint tokens."""
        text = [
            "one two one two three four five six five six",
            "six seven eight eight nine ten ten ten ten",
        ]
        metric = Yules_I()
        calculated_metrics = metric.compute({}, Predictions({"values": text}))
        self.assertAlmostEqual(calculated_metrics[f"yules_i"], 2.857)

    def test_yules_i_identical_tokens(self):
        """Tests for Yules_I with identical tokens (low diversity)."""
        text = [
            "token token token token token token token token token token token token token",
            "token token token token token token token token token token token token token",
        ]
        metric = Yules_I()
        calculated_metrics = metric.compute({}, Predictions({"values": text}))
        self.assertAlmostEqual(calculated_metrics[f"yules_i"], 0.001)


if __name__ == "__main__":
    unittest.main()
