#!/usr/bin/env python3

from .metric import ReferencedMetric
from datasets import load_metric
import numpy as np


class BERTScore(ReferencedMetric):
    """BERTScore uses the tiny checkpoint for efficient CPU runtime."""
    def __init__(self):
        """Load the BERT checkpoint into memory."""
        self.metric = load_metric('bertscore')

    def _make_serializable(self, score_entry):
        """Convert from tensor object to list of floats."""
        return [float(score) for score in score_entry.numpy()]

    def compute(self, predictions, references):
        """Run BERTScore."""
        self.metric.add_batch(
                predictions=predictions.untokenized,
                references=references.untokenized)
        # Use language-appropriate scorer.
        score = self.metric.compute(lang=predictions.language.alpha_2, model_type ='distilbert-base-uncased')
        score['precision'] = np.mean(self._make_serializable(score['precision']))
        score['recall'] = np.mean(self._make_serializable(score['recall']))
        score['f1'] = np.mean(self._make_serializable(score['f1']))
        return {'bertscore': score}
