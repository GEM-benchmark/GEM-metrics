from .metric import ReproReferencedMetric

from repro.models.kane2020 import NUBIA as _NUBIA
from typing import List


class NUBIA(ReproReferencedMetric):
    def __init__(self):
        metric = _NUBIA()
        super().__init__(metric)

    def _postprocess(self, score_dicts: List) -> List:
        # The Repro version also contains other outputs from
        # NUBIA. Here we just select the score
        return [{"nubia": scores["nubia"]["nubia_score"]} for scores in score_dicts]
