#!/usr/bin/env python3

import numpy as np
from .metric import SourceAndReferencedMetric
from questeval.questeval_metric import Questeval


class QuestEval(SourceAndReferencedMetric):
    def __init__(self):
        # Default values
        self.task = "summarization"
        self.language = "en"

        self.metric = Questeval(
            task=self.task,
            language=self.language,
            isCuda=True,
        )

    def compute(self, predictions, references, sources):
        # Not using references for now, but we will in future.
        if predictions.task != self.task or predictions.language.alpha_2 != self.language:
            self.task = predictions.task
            self.language = predictions.language
            self.metric = Questeval(
                task=predictions.task,
                language=predictions.language,
                isCuda=True,
            )

        # TODO: For better code comprehension, not batched for now. (maybe in future)
        # TODO: Use cached version (maybe in future)
        scores = [
            self.metric.compute_all(p, s)["scores"]
            for p, s in zip(predictions.untokenized, sources.untokenized)
        ]

        return {
            'questeval': {
                'precision': np.mean([s["precision"] for s in scores]),
                'recall': np.mean([s["recall"] for s in scores]),
                'fscore': np.mean([s["fscore"] for s in scores]),
            }
        }
