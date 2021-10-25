"""Test class for metrics that use both source and a reference.
"""
import unittest
from gem_metrics.texts import Predictions, References, Sources
from tests.inputs import TestData
from tests.test_referenced import TestReferencedMetric
from tests.utils import assertDeepAlmostEqual


class TestSourcedAndReferencedMetric(TestReferencedMetric):

    def get_calculated_metrics(self, references: References, predictions: Predictions):
        calculated_metrics = self.metric.compute(
            sources=TestData.sources, references=references, predictions=predictions, cache={}
        )
        calculated_metrics = self.postprocess_metrics(calculated_metrics)
        return calculated_metrics

if __name__ == "__main__":
    unittest.main()
