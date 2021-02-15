"""Test class for metrics that use both source and a reference.
"""
import unittest
from gem_metrics.texts import Predictions, References, Sources
from tests.inputs import TestData
from tests.utils import assertDeepAlmostEqual


class TestSourcedAndReferencedMetric(object):

    def setUp(self):
        self.metric_name = None
        self.metrics_keys_to_ignore = None
        self.true_results_basic = None
        self.true_results_identical_pred_ref = None
        self.true_results_mismatched_pred_ref = None
        self.true_results_empty_pred = None
        self.test_precision = 2

    def test_metric(self):
        calculated_metrics = self.get_calculated_metrics(
            sources=TestData.sources, references=TestData.references, predictions=TestData.predictions)
        assertDeepAlmostEqual(self, calculated_metrics, self.true_results_basic, places=self.test_precision)

    def test_metric_identical_pred_ref(self):
        """Tests for identical predictions and references
        """
        calculated_metrics = self.get_calculated_metrics(
            sources=TestData.sources, references=TestData.references, predictions=TestData.identical_predictions)
        assertDeepAlmostEqual(self, calculated_metrics, self.true_results_identical_pred_ref, places=self.test_precision)

    def test_metric_mismatched_pred_ref(self):
        """Tests for completely dissimilar predictions and references
        """
        calculated_metrics = self.get_calculated_metrics(
            sources=TestData.sources, references=TestData.references, predictions=TestData.reversed_predictions)
        assertDeepAlmostEqual(self, calculated_metrics, self.true_results_mismatched_pred_ref, places=self.test_precision)

    def test_metric_mismatched_empty_tgt(self):
        """Tests for empty target
        """
        calculated_metrics = self.get_calculated_metrics(
            sources=TestData.sources, references=TestData.references, predictions=TestData.empty_predictions)
        assertDeepAlmostEqual(self, calculated_metrics, self.true_results_empty_pred, places=self.test_precision)

    def get_calculated_metrics(self, sources: Sources, references: References, predictions: Predictions):
        calculated_metrics = self.metric.compute(sources=sources, references=references, predictions=predictions)
        if self.metrics_keys_to_ignore:
            for metric in self.metrics_keys_to_ignore:
                for key in self.metrics_keys_to_ignore[metric]:
                    calculated_metrics[metric].pop(key)
        return calculated_metrics


if __name__ == '__main__':
    unittest.main()
