#!/usr/bin/env python3


# Data holder classes
from .texts import Predictions, References, Submission
from typing import Optional

# Metric implementations
from .meteor import Meteor
from .bleu import BLEU
from .rouge import ROUGE
from .msttr import MSTTR
from .ngrams import NGramStats

# Lists of metrics to use
# TODO make this populate automatically based on imports
REFERENCED_METRICS = [BLEU, Meteor, ROUGE]
REFERENCELESS_METRICS = [MSTTR, NGramStats]


def compute(outs: Predictions, refs: Optional[References]) -> dict:
    """Main metrics computation routine. Expects a Predictions and a References object, holding
    system outputs and corresponding references (References may be None -- only referenceless metrics
    are computed in such a case).
    Returns a dict with the results.
    """
    # initialize values storage
    values = {'predictions_file': outs.filename,
              'N': len(outs)}

    # compute referenceless metrics
    for metric_class in REFERENCELESS_METRICS:
        metric = metric_class()
        values.update(metric.compute(outs))

    # compute ref-based metrics
    if refs is not None:
        values['references_file'] = refs.filename
        for metric_class in REFERENCED_METRICS:
            metric = metric_class()
            values.update(metric.compute(outs, refs))
    return values


def load_references(dataset_name: str) -> Optional[References]:
    """Load a file with references for a standard GEM dataset."""
    # TODO not implemented yet
    return None
