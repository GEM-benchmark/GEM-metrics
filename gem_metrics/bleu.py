#!/usr/bin/env python3

from .metric import ReferencedMetric
from .impl import pymteval


class BLEU(ReferencedMetric):
    """BLEU uses a Python implementation from E2E-metrics (https://github.com/tuetschek/e2e-metrics),
    which should be compatible with SacreBLEU or the original mteval-v13a script used by WMT (up to
    4th digit -- there is some slight smoothing to get around edge cases, such as no matching n-grams)."""

    def compute(self, predictions, references):
        bleu = pymteval.BLEUScore()
        for refs, pred in zip(references.untokenized, predictions.untokenized):
            bleu.append(pred, refs)
        return {'bleu': bleu.score()}
