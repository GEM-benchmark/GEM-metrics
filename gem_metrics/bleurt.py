#!/usr/bin/env python3
from .data import ensure_download
from .metric import ReferencedMetric
from .texts import Predictions, References

from bleurt import score
import numpy as np
import tensorflow as tf
from typing import Dict

class BLEURT(ReferencedMetric):
    """BLEURT uses the base checkpoint for efficient runtime."""

    def __init__(self, checkpoint_path="bleurt-base-128"):
        """Load the BLEURT checkpoint into memory."""
        # Moved to initialize to support caching without initialization.
        self.checkpoint_path = checkpoint_path

    def _initialize(self):
        ckpt_path = ensure_download(
            "models",
            self.checkpoint_path,
            f"https://storage.googleapis.com/bleurt-oss/{self.checkpoint_path}.zip",
        )
        self.metric = score.BleurtScorer(ckpt_path)

    def compute(self, cache, predictions: Predictions, references: References) -> Dict:
        """Compute the BLEURT score. Multi-ref will be averaged."""
        # Use untokenized here since the module uses its own tokenizer.
        if isinstance(references.untokenized[0], list):
            # For multi-reference data, compute micro-average.
            scores = []
            for pred, refs in zip(predictions.untokenized, references.untokenized):
                pred_repeated = [pred] * len(refs)
                example_scores = self.metric.score(
                    references=refs, candidates=pred_repeated
                )
                scores.append(np.mean(example_scores))
        else:
            scores = self.metric.score(
                references=references.untokenized,
                candidates=predictions.untokenized,
                batch_size=64,
            )

        formatted_scores = {}
        for sc, pred_id in zip(scores, predictions.ids):
            formatted_score = {"bleurt": sc}
            formatted_scores[pred_id] = formatted_score
            if cache is not None:
                cache_key = (self.__class__.__name__, predictions.filename, pred_id)
                cache[cache_key] = formatted_score

        return formatted_scores
