#!/usr/bin/env python3

class ReferencedMetric:
    """Base class for all referenced metrics."""

    def compute(self, predictions, references):
        pass

   
class SourceAndReferencedMetric:
    """Base class for all metrics that require source and reference sentences."""

    def compute(self, predictions, references, sources):
        pass


class ReferencelessMetric:
    """Base class for all referenceless metrics."""

    def compute(self, predictions):
        pass
