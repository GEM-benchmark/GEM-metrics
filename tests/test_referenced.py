"""Test class for referenced tests
"""
import unittest
from tests.inputs import TestData
from tests.utils import assertDeepAlmostEqual


class ReferencedMetricTest(object):

    def test_metric(self):
        """Tests for identical predictions and references
        """

        calculated_metrics = self.metric.compute(
            references=TestData.references, predictions=TestData.predictions)
        assertDeepAlmostEqual(self, calculated_metrics, self.expected_result_basic, places=2)

    def test_metric_identical_pred_ref(self):
        """Tests for identical predictions and references
        """
        calculated_metrics = self.metric.compute(
            references=TestData.references, predictions=TestData.identical_predictions)
        assertDeepAlmostEqual(self, calculated_metrics, self.expected_result_identical_pred_ref)

    def test_metric_mismatched_pred_ref(self):
        """Tests for completely dissimilar predictions and references
        """
        calculated_metrics = self.metric.compute(references=TestData.references,
                                                      predictions=TestData.reversed_predictions)
        assertDeepAlmostEqual(self, calculated_metrics, self.expected_result_mismatched_pred_ref, places=2)

    def test_metric_mismatched_empty_tgt(self):
        """Tests for empty target
        """
        calculated_metrics = self.metric.compute(
            references=TestData.references, predictions=TestData.empty_predictions)
        assertDeepAlmostEqual(self, calculated_metrics, self.expected_result_empty_pred_ref)


if __name__ == '__main__':
    unittest.main()
