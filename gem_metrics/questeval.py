#!/usr/bin/env python3

import numpy as np
from .metric import SourceAndReferencedMetric
from questeval.questeval_metric import QuestEval as QuestEvalMetric
from logzero import logger


class QuestEval(SourceAndReferencedMetric):
    def __init__(self):
        # Default values
        self.task = "summarization"
        self.language = "en"
        self._this_task_is_available = True

        self.metric = QuestEvalMetric(
            task=self.task,
            language=self.language,
            isCuda=True,
        )

    def compute(self, predictions, references, sources):
        # TODO: For better code comprehension, not batched for now. (maybe in future)
        # TODO: Use cached version (maybe in future)
        # TODO: only mono reference for now.
        # Not using references for now, but we will in future.
        if predictions.task != self.task or predictions.language.alpha_2 != self.language:
            self.task = predictions.task
            self.language = predictions.language

            # Checking if the task is available
            task = predictions.task
            self._this_task_is_available = True
            if self.task not in self.metric.AVAILABLE_TASKS:
                self._this_task_is_available = False
                task = "text2text"
                logger.warning("This task is not available, QuestMetric is using the general text2text models.")

            self.metric = QuestEvalMetric(
                task=task,
                language=predictions.language.alpha_2,
                isCuda=True,
            )

        # If the task is not available, then we give references instead of sources
        local_sources, local_references = sources.untokenized, [[None]] * len(sources.untokenized)
        if self._this_task_is_available is False:
            local_sources, local_references = [None] * len(references.untokenized), references.untokenized

        # Computing scores
        scores = [
            self.metric.compute_all(p, source=s, reference=r[0])["scores"]
            for p, s, r in zip(predictions.untokenized, local_sources, local_references)
        ]

        return {
            'questeval': {
                'precision': np.mean([s["precision"] for s in scores]),
                'recall': np.mean([s["recall"] for s in scores]),
                'f1': np.mean([s["fscore"] for s in scores]),
            }
        }
