#!/usr/bin/env python3

from .metric import ReferencedMetric
from datasets import load_metric
import numpy as np


class BLEURT(ReferencedMetric):
    """BLEURT uses the tiny checkpoint for efficient CPU runtime."""
    def __init__(self, checkpoint_path="bleurt-base-128"):
        """Load the BLEURT checkpoint into memory."""
        self.metric = load_metric('bleurt', checkpoint_path)

    def compute(self, predictions, references):
        """Compute the BLEURT score. Multi-ref will be averaged."""
        # Use untokenized here since the module uses its own tokenizer.
        if isinstance(references.untokenized[0], list):
            # For multi-reference data, compute micro-average.
            scores = []
            for pred, refs in zip(predictions.untokenized,
                                  references.untokenized):
                pred_repeated = [pred] * len(refs)
                self.metric.add_batch(
                    predictions=pred_repeated,
                    references=refs)
                scores.append(np.mean(self.metric.compute()['scores']))
        else:
            self.metric.add_batch(
                predictions=predictions.untokenized,
                references=references.untokenized)
            scores = self.metric.compute()['scores']
        return {'bleurt': np.mean(scores)}

