#!/usr/bin/env python3


class Metric:
    """Base class for all metrics."""

    def compute(self, predictions, references):
        pass
