import unittest
import gem_metrics.rouge
from tests.test_referenced import TestReferencedMetric


class TestRouge(TestReferencedMetric, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.metric = gem_metrics.rouge.ROUGE()
        self.true_results_basic = {
            "rouge1": {
                "precision": round(0.673015873015873, 5),
                "recall": round(0.5777777777777778, 5),
                "fmeasure": round(0.6203949307397583, 5),
            },
            "rouge2": {
                "precision": round(0.36130536130536123, 5),
                "recall": round(0.32051282051282054, 5),
                "fmeasure": round(0.3395061728395062, 5),
            },
            "rougeL": {
                "precision": round(0.5785714285714286, 5),
                "recall": round(0.5063492063492063, 5),
                "fmeasure": round(0.5391983495431771, 5),
            },
            "rougeLsum": {
                "precision": round(0.5785714285714286, 5),
                "recall": round(0.5063492063492063, 5),
                "fmeasure": round(0.5391983495431771, 5),
            },
        }
        self.true_results_identical_pred_ref = {
            "rouge1": {
                "precision": 1.0,
                "recall": 1.0,
                "fmeasure": 1.0,
            },
            "rouge2": {"precision": 1.0, "recall": 1.0, "fmeasure": 1.0},
            "rougeL": {"precision": 1.0, "recall": 1.0, "fmeasure": 1.0},
            "rougeLsum": {"precision": 1.0, "recall": 1.0, "fmeasure": 1.0},
        }
        self.true_results_mismatched_pred_ref = {
            "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rougeLsum": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        }
        self.true_results_empty_pred = {
            "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rougeLsum": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        }


if __name__ == "__main__":
    unittest.main()
