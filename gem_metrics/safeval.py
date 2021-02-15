#!/usr/bin/env python3

from .metric import SourcedMetric
import numpy as np
from safeval.safeval_metric import QG_policy_scorer


class SAFEval(SourcedMetric):
    def __init__(self):
        # TODO: need to define default values
        self.metric = QG_policy_scorer(
            limit_sent=5,
            answer_types=['NER', 'NOUN'],
            qg_beam_size=20,
            QG_top_p=1,
            lambda_penalty=-0.05,
            isCuda=True
            # lang="en",
            # task="summarization",
        )
        self.task = "summarization"
        self.language = "en"

    def compute(self, predictions, sources):
        if predictions.task != self.task and predictions.language != self.language:
            # TODO: change metric in case of task change
            # self.metric = self.metric
            print("We need to change QA/QG models")

        # TODO: For better code comprehension, not batched for now.
        scores = [
            self.metric.compute_all(p, s)
            for p, s in zip(predictions.untokenized, sources.untokenized)
        ]

        # TODO: Add other sub metrics.
        return {
            'safeval': {
                'precision': 0,
                'recall': 0,
                'mean': np.mean(scores)
            }
        }
