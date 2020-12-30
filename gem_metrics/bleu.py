#!/usr/bin/env python3

from .metric import Metric
from .external import pymteval


class BLEU(Metric):

    def compute(self, predictions, references):
        bleu = pymteval.BLEUScore()
        for refs, pred in zip(references.untokenized, predictions.untokenized):
            bleu.append(pred, refs)
        return {'bleu': bleu.score()}
