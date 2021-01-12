#!/usr/bin/env python3

from .metric import ReferencedMetric
from .impl.meteor import PyMeteorWrapper


class Meteor(ReferencedMetric):
    """METEOR uses the original Java Meteor-1.5 implementation with a wrapper adapted from
    MSCOCO/E2E-metrics."""

    def compute(self, predictions, references):
        m = PyMeteorWrapper('en')
        # ignore individual sentence scores
        meteor, _ = m.compute_score(predictions.untokenized, references.untokenized)
        return {'meteor': meteor}

