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
        )

    def compute(self, predictions, sources):
        # TODO: For better code comprehension, not batched for now.
        scores = [
            self.metric.compute_all(p, s)
            for p, s in zip(predictions.untokenized, sources.untokenized)
        ]
        return {'safeval': np.mean(scores)}
