#!/usr/bin/env python3

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
        )

    def compute(self, predictions, references, sources):
        # If task or language is different, we must change QA / QG models for questeval
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
            )

        # If the task was not available, then we pass references instead of sources
        local_sources, local_references = sources.untokenized, [[None]] * len(sources.untokenized)
        if self._this_task_is_available is False:
            local_sources, local_references = [None] * len(references.untokenized), references.untokenized

        # Computing scores through one batched step
        scores = self.metric.corpus_questeval(
            hypothesis=predictions.untokenized,
            sources=local_sources,
            list_references=local_references,
        )

        return {
            'questeval': {
                "f1": scores["corpus_score"]
            }
        }
