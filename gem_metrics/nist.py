#!/usr/bin/env python3

from .metric import ReferencedMetric
from .impl.pymteval import NISTScore


class NIST(ReferencedMetric):
    """NIST from e2e-metrics."""

    def compute(self, predictions, references):

        nist = NISTScore()
        for pred, refs in zip(predictions.untokenized, references.untokenized):
            nist.append(pred, refs)
        return {'nist': nist.score()}
