"""Test class for metrics that use both source and a reference.
"""
import unittest
from tests.inputs import TestData
from tests.utils import assertDeepAlmostEqual


class TestSourcedAndReferencedMetric(object):

    test_precision = 2
    def test_metric(self):
        """Tests for identical predictions and references
        """

        calculated_metrics = self.metric.compute(
            sources=TestData.sources, references=TestData.references, predictions=TestData.predictions)
        assertDeepAlmostEqual(self, calculated_metrics, self.expected_result_basic, places=TestSourcedAndReferencedMetric.test_precision)

    def test_metric_identical_pred_ref(self):
        """Tests for identical predictions and references
        """
        calculated_metrics = self.metric.compute(
            sources=TestData.sources, references=TestData.references, predictions=TestData.identical_predictions)
        assertDeepAlmostEqual(self, calculated_metrics, self.expected_result_identical_pred_ref, places=TestSourcedAndReferencedMetric.test_precision)

    def test_metric_mismatched_pred_ref(self):
        """Tests for completely dissimilar predictions and references
        """
        calculated_metrics = self.metric.compute(
            sources=TestData.sources, references=TestData.references, predictions=TestData.reversed_predictions)
        assertDeepAlmostEqual(self, calculated_metrics, self.expected_result_mismatched_pred_ref, places=TestSourcedAndReferencedMetric.test_precision)

    def test_metric_mismatched_empty_tgt(self):
        """Tests for empty target
        """
        calculated_metrics = self.metric.compute(
            sources=TestData.sources, references=TestData.references, predictions=TestData.empty_predictions)
        assertDeepAlmostEqual(self, calculated_metrics, self.expected_result_empty_pred_ref, places=TestSourcedAndReferencedMetric.test_precision)


if __name__ == '__main__':
    unittest.main()
