#!/usr/bin/env python3

from .metric import ReferencedMetric
from .impl.meteor import PyMeteorWrapper
from logzero import logger


class Meteor(ReferencedMetric):
    """METEOR uses the original Java Meteor-1.5 implementation with a wrapper adapted from
    MSCOCO/E2E-metrics."""

    def compute(self, predictions, references):
        try:
            m = PyMeteorWrapper(predictions.language.alpha_2)
        except Exception as e:
            logger.warn(f'Cannot run Meteor -- Skipping: {str(e)}')
            return {'meteor': None}
        # ignore individual sentence scores
        meteor, _ = m.compute_score(predictions.untokenized, references.untokenized)
        return {'meteor': meteor}

