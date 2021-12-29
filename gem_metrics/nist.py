#!/usr/bin/env python3

from .texts import Predictions, References
from .metric import ReferencedMetric
from .impl.pymteval import NISTScore

from typing import Dict

class NIST(ReferencedMetric):
    """NIST from e2e-metrics."""

    def support_caching(self):
        # NIST is corpus-level, so individual examples can't be aggregated.
        return False

    def compute(self, cache, predictions: Predictions, references: References) -> Dict:
        nist = NISTScore()
        for pred, refs in zip(predictions.untokenized, references.untokenized):
            nist.append(pred, refs)
        return {"nist": nist.score()}
