"""Test class for referenced tests
"""
from gem_metrics.texts import Predictions, References
import unittest
from tests.inputs import TestData
from tests.utils import assertDeepAlmostEqual


class TestReferencedMetric(object):

    def setUp(self):
        self.metric_name = None
        self.metrics_keys_to_ignore = None
        self.expected_result_basic = None
        self.expected_result_identical_pred_ref = None
        self.expected_result_mismatched_pred_ref = None
        self.expected_result_empty_pred_ref = None

    def test_metric(self):
        """Tests for identical predictions and references
        """

        calculated_metrics = self.get_calculated_metrics(
            references=TestData.references, predictions=TestData.predictions)
        assertDeepAlmostEqual(self, calculated_metrics,
                              self.expected_result_basic, places=2)

    def test_metric_identical_pred_ref(self):
        """Tests for identical predictions and references
        """
        calculated_metrics = self.get_calculated_metrics(
            references=TestData.references, predictions=TestData.identical_predictions)
        assertDeepAlmostEqual(self, calculated_metrics,
                              self.expected_result_identical_pred_ref)

    def test_metric_mismatched_pred_ref(self):
        """Tests for completely dissimilar predictions and references
        """
        calculated_metrics = self.get_calculated_metrics(references=TestData.references,
                                                         predictions=TestData.reversed_predictions)
        assertDeepAlmostEqual(self, calculated_metrics,
                              self.expected_result_mismatched_pred_ref, places=2)

    def test_metric_empty_tgt(self):
        """Tests for empty target
        """
        calculated_metrics = self.get_calculated_metrics(
            references=TestData.references, predictions=TestData.empty_predictions)
        assertDeepAlmostEqual(self, calculated_metrics,
                              self.expected_result_empty_pred_ref)

    def get_calculated_metrics(self, references: References, predictions: Predictions):
        calculated_metrics = self.metric.compute(
            references=references, predictions=predictions)
        if self.metrics_keys_to_ignore:
            for metric in self.metrics_keys_to_ignore:
                for key in self.metrics_keys_to_ignore[metric]:

                    calculated_metrics[metric].pop(key)
        return calculated_metrics


if __name__ == '__main__':
    unittest.main()
