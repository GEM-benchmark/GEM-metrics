"""Test class for referenced metrics.
"""
from gem_metrics.texts import Predictions, References
import unittest
from tests.inputs import TestData
from tests.utils import assertDeepAlmostEqual


class TestReferencedMetric(object):
    def setUp(self):
        self.metric_name = None
        self.metrics_keys_to_ignore = None
        self.true_results_basic = None
        self.true_results_identical_pred_ref = None
        self.true_results_mismatched_pred_ref = None
        self.true_results_empty_pred = None
        self.test_precision = 2

    def _run_test(self, references: References, predictions: Predictions, true_results):
        # some metrics such as bertscore rely on this attribute for metric computation
        predictions.ids = [str(i) for i in range(len(predictions))]
        calculated_metrics = self.get_calculated_metrics(
            references=references, predictions=predictions
        )
        assertDeepAlmostEqual(
            self,
            calculated_metrics,
            true_results,
            places=self.test_precision,
        )

    def test_metric(self):
        self._run_test(
            TestData.references, TestData.predictions, self.true_results_basic
        )

    def test_metric_identical_pred_ref(self):
        """Tests for identical predictions and references"""
        self._run_test(
            TestData.references,
            TestData.identical_predictions,
            self.true_results_identical_pred_ref,
        )

    def test_metric_mismatched_pred_ref(self):
        """Tests for completely dissimilar predictions and references"""
        self._run_test(
            TestData.references,
            TestData.reversed_predictions,
            self.true_results_mismatched_pred_ref,
        )

    def test_metric_empty_tgt(self):
        """Tests for empty target"""
        self._run_test(
            TestData.references,
            TestData.empty_predictions,
            self.true_results_empty_pred,
        )

    def get_calculated_metrics(self, references: References, predictions: Predictions):
        calculated_metrics = self.metric.compute(
            references=references, predictions=predictions, cache={}
        )
        calculated_metrics = self.postprocess_metrics(calculated_metrics)
        return calculated_metrics

    def postprocess_metrics(self, calculated_metrics):
        if self.metric.support_caching():
            calculated_metrics = list(calculated_metrics.values())
            calculated_metrics = self.metric._aggregate_scores(calculated_metrics)

        if self.metrics_keys_to_ignore:
            for metric in self.metrics_keys_to_ignore:
                for key in self.metrics_keys_to_ignore[metric]:
                    calculated_metrics[metric].pop(key)

        return calculated_metrics


if __name__ == "__main__":
    unittest.main()
