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
    """Main metrics computation routine for a single dataset.
    Expects a Predictions and a References object, holding system outputs and corresponding
    references (References may be None -- only referenceless metrics are computed in such a case).
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


def process_submission(outs: Submission, refs: Optional[dict]) -> dict:
    """Process a (potentially) multi-dataset submission. Expects a Submission object
    holding all the predictions, and potentially references in a dictionary keyed by
    dataset name, paralleling the datasets in the submission. If no references are
    given and the dataset names correspond to GEM task datasets, default references are used.
    Returns a dict keyed by dataset names, containing the dicts for each dataset's results.
    """
    values = {}
    for dataset in outs.datasets:
        outs_ds = outs.predictions_for(dataset)
        # use default reference files if no custom ones are provided
        refs_ds = refs[dataset] if refs else load_references(dataset)
        if refs:
            assert(len(refs_ds) == len(outs_ds))
        values[dataset] = compute(outs_ds, refs_ds)
    return values


def load_references(dataset_name: str) -> Optional[References]:
    """Load a file with references for a standard GEM dataset."""
    # TODO not implemented yet
    return None
