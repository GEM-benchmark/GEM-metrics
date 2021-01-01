#!/usr/bin/env python3


class ReferencedMetric:
    """Base class for all referenced metrics."""

    def compute(self, predictions, references):
        pass


class ReferencelessMetric:
    """Base class for all referenceless metrics."""

    def compute(self, predictions):
        pass
