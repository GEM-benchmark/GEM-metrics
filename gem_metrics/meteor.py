#!/usr/bin/env python3

from .metric import ReferencedMetric
from .impl.meteor import PyMeteorWrapper
from logzero import logger


class Meteor(ReferencedMetric):
    """METEOR uses the original Java Meteor-1.5 implementation with a wrapper adapted from
    MSCOCO/E2E-metrics."""

    def support_caching(self):
        # METEOR is corpus-level, so individual examples can't be aggregated.
        # While individual scores can be computed, the overall score is different.
        return False

    def compute(self, cache, predictions, references):
        try:
            m = PyMeteorWrapper(predictions.language.alpha_2)
        except Exception as e:
            logger.warn(f"Cannot run Meteor -- Skipping: {str(e)}")
            return {"meteor": None}
        # ignore individual sentence scores
        try:
            meteor, _ = m.compute_score(predictions.untokenized, references.untokenized)
        except BrokenPipeError:
            logger.warn("METEOR FAILED TO COMPUTE.")
            meteor = -99
        return {"meteor": meteor}
