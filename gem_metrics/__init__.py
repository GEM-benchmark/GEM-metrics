#!/usr/bin/env python3

from logzero import logger
from typing import Optional

# Data holder classes
from .texts import Predictions, References, Submission

# Metric implementations
from .meteor import Meteor
from .bertscore import BERTScore
from .bleu import BLEU
from .bleurt import BLEURT
from .rouge import ROUGE
from .msttr import MSTTR
from .ngrams import NGramStats
from .data import ensure_download

# Lists of metrics to use
# TODO make this populate automatically based on imports
REFERENCED_METRICS = [BERTScore, BLEU, BLEURT, Meteor, ROUGE]
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
    values = {'submission_name': outs.name}
    for dataset in outs.datasets:
        outs_ds = outs.predictions_for(dataset)
        # use default reference files if no custom ones are provided
        refs_ds = refs[dataset] if refs else load_references(dataset)
        if refs:
            assert(len(refs_ds) == len(outs_ds))
        values[dataset] = compute(outs_ds, refs_ds)
    return values


# URLs to download standard references from
_DATASET_REFERENCES_URLS = {
    'asset_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/asset_test.json',
    'asset_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/asset_val.json',
    'common_gen_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/common_gen_val.json',
    'cs_restaurants_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/cs_restaurants_test.json',
    'cs_restaurants_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/cs_restaurants_val.json',
    'e2e_nlg_cleaned_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/e2e_nlg_cleaned_test.json',
    'e2e_nlg_cleaned_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/e2e_nlg_cleaned_val.json',
    'turk_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/turk_test.json',
    'turk_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/turk_val.json',
    'webnlg_en_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/webnlg_en_val.json',
    'webnlg_en_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/webnlg_en_test.json',
    'webnlg_ru_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/webnlg_ru_val.json',
    'webnlg_ru_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/webnlg_ru_test.json',
}


def load_references(dataset_name: str) -> Optional[References]:
    """Load a file with references for a standard GEM dataset (attempt download), return None if not present."""
    if dataset_name in _DATASET_REFERENCES_URLS:
        try:
            dataset_file = ensure_download('references', dataset_name + '.json', _DATASET_REFERENCES_URLS[dataset_name])
            return References(dataset_file)
        except Exception as e:
            logger.warn(f'Could not download references for {dataset_name}: {str(e)}')
            return None
    return None
