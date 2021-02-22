#!/usr/bin/env python3

from .metric import SourcedMetric
from safeval.safeval_metric import QGPolicyScorer


class SAFEval(SourcedMetric):
    def __init__(self):
        # TODO: need to define default values depending on each task (at the end)
        self.metric = QGPolicyScorer(
            limit_sent=5,
            answer_types=['NER', 'NOUN'],
            need_policy=True,
            qg_beam_size=1,
            QG_top_p=1,
            lambda_penalty=-0.05,
            isCuda=True,
            language="en",
            task="summarization",
        )
        self.language = "en"
        self.task = "summarization"

    def compute(self, predictions, sources):
        if predictions.task != self.task or predictions.language.alpha_2 != self.language:
            self.metric = QGPolicyScorer(
                limit_sent=5,
                answer_types=['NER', 'NOUN'],
                need_policy=True,
                qg_beam_size=1,
                QG_top_p=1,
                lambda_penalty=-0.05,
                isCuda=True,
                language=predictions.language,
                task=predictions.task,
            )
            self.language = predictions.language
            self.task = predictions.task

        # TODO: For better code comprehension, not batched for now. (maybe in future)
        # TODO: Add other sub metrics (precision, recall...). (at the end)
        score = self.metric.compute_all_cached(
            predictions.untokenized, sources.untokenized,
            source_filename=sources.filename
        )

        return {
            'safeval': {
                'precision': 0,
                'recall': 0,
                'mean': score
            }
        }
