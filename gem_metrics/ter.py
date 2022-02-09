#!/usr/bin/env python3

from .metric import ReferencedMetric
from .texts import Predictions, References

from typing import Dict
from sacrebleu.metrics import TER as _TER
from itertools import zip_longest


class TER(ReferencedMetric):
    """Translation error rate (TER) from SacreBLEU."""

    def __init__(self, case_sensitive: bool = False):
            self.metric = _TER(case_sensitive=case_sensitive)

    def support_caching(self):
        # corpus-level, so individual examples can't be aggregated.
        return False

    def compute(self, cache, predictions: Predictions, references: References) -> Dict:
        ref_streams = list(zip_longest(*references.untokenized))
        ter = self.metric.corpus_score(
            predictions.untokenized, ref_streams)
        return {"ter": round(ter.score, 5)}
