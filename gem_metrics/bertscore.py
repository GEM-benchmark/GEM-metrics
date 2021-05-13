#!/usr/bin/env python3

from .metric import ReferencedMetric
from datasets import load_metric
import numpy as np


class BERTScore(ReferencedMetric):
    """BERTScore uses the tiny checkpoint for efficient CPU runtime."""

    def __init__(self):
        """Load the BERT checkpoint into memory."""
        self.metric = load_metric("bertscore", batch_size=64)

    def _make_serializable(self, score_entry):
        """Convert from tensor object to list of floats."""
        return [float(score) for score in score_entry]

    def compute(self, cache, predictions, references):
        """Run BERTScore."""
        self.metric.add_batch(
            predictions=predictions.untokenized, references=references.untokenized
        )
        # Use language-appropriate scorer.
        score = self.metric.compute(
            lang=predictions.language.alpha_2, model_type="distilbert-base-uncased"
        )
        
        precisions = self._make_serializable(score["precision"])
        recalls = self._make_serializable(score["recall"])
        f1s = self._make_serializable(score["f1"])
        
        scores = {}
        for pred_id, prec, rec, f1 in zip(
            predictions.ids, precisions, recalls, f1s):
            score_obj = {
                "bertscore": {
                    "precision": prec,
                    "recall": rec,
                    "f1": f1
                    }
                }
            
            # Write to cache if not None.
            if cache is not None:
                cache_key = (self.__class__.__name__, predictions.filename, pred_id)
                cache[cache_key] = score_obj
            scores[pred_id] = score_obj

        return scores
        # score["precision"] = np.mean()
        # score["recall"] = np.mean(self._make_serializable(score["recall"]))
        # score["f1"] = np.mean(self._make_serializable(score["f1"]))
        # return {"bertscore": score}
