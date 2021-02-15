#!/usr/bin/env python3

from logzero import logger
from typing import Optional, Dict, List

# Data holder classes
from .texts import Predictions, References, Sources, Submission

# Metric implementations
from .meteor import Meteor
from .bertscore import BERTScore
from .bleu import BLEU
from .bleurt import BLEURT
from .rouge import ROUGE
from .local_recall import LocalRecall
from .msttr import MSTTR
from .ngrams import NGramStats
from .data import ensure_download
from .sari import SARI
from .safeval import SAFEval


def metric_list_to_metric_dict(metric_list: List[str]) -> Dict[str, List]:
    '''
    Function that converts a list of strings corresponding to the metric names into a dictionary with three keys,
    referenced_metrics, sourced_metrics, referenceless_metrics, and sourced_and_referenced_metrics, which are
    populated by the actual metrics class.
    '''
    # convert to set in case there are repeats
    metric_list = list(set(metric_list))

    metric_name_to_metric_class = {
        'bertscore': BERTScore,
        'bleu': BLEU,
        'bleurt': BLEURT,
        'local_recall': LocalRecall,
        'meteor': Meteor,
        'rouge': ROUGE,
        'msttr': MSTTR,
        'ngram': NGramStats,
        'sari': SARI,
        'safeval': SAFEval,
    }

    metric_name_to_metric_type = {
        'bertscore': 'referenced',
        'bleu': 'referenced',
        'bleurt': 'referenced',
        'local_recall': 'referenced',
        'meteor': 'referenced',
        'rouge': 'referenced',
        'msttr': 'referenceless',
        'ngram': 'referenceless',
        'sari': 'sourced_and_referenced',
        'safeval': 'sourced',
    }

    referenced_list, sourced_list, referenceless_list, sourced_and_referenced_list = [], [], [], []

    for metric_name in metric_list:
        metric_class = metric_name_to_metric_class[metric_name]
        metric_type = metric_name_to_metric_type[metric_name]
        if metric_type == 'referenced':
            referenced_list.append(metric_class)
        elif metric_type == 'sourced':
            sourced_list.append(metric_class)
        elif metric_type == 'referenceless':
            referenceless_list.append(metric_class)
        elif metric_type == 'sourced_and_referenced':
            sourced_and_referenced_list.append(metric_class)
        else:
            raise NotImplementedError(f'{metric_type} is not one of [referenced, referenceless, sourced_and_referenced]. Please check the metric_name_to_metric_type dict.')

    metric_dict = {
        'referenced_metrics': referenced_list,
        'sourced_metrics': sourced_list,
        'referenceless_metrics': referenceless_list,
        'sourced_and_referenced_metrics': sourced_and_referenced_list,
    }

    return metric_dict


def compute(outs: Predictions, refs: Optional[References] = None, srcs: Optional[Sources] = None, metrics_dict: Dict[str, List] = None) -> Dict:
    """Main metrics computation routine for a single dataset.
    Expects a Predictions and a References object, holding system outputs and corresponding
    references (References may be None -- only referenceless metrics are computed in such a case).
    metrics_dict is a dictionary with three keys: referenced_metrics, sourced_metrics, referenceless_metrics,
    and sourced_and_referenced_metrics. Each of those keys' values are a List of the specific metrics.
    Returns a dict with the results.
    """
    # initialize values storage
    values = {'predictions_file': outs.filename,
              'N': len(outs)}

    # compute referenceless metrics
    for metric_class in metrics_dict['referenceless_metrics']:
        metric = metric_class()
        values.update(metric.compute(outs))

    # compute ref-based metrics
    if refs is not None:
        if len(refs) != len(outs):
            raise ValueError(f'Incorrect length for data "{outs.filename}" -- outputs: {len(outs)} vs. references: {len(refs)}')
        values['references_file'] = refs.filename
        for metric_class in metrics_dict['referenced_metrics']:
            metric = metric_class()
            values.update(metric.compute(outs, refs))

    # compute src-based metrics
    if srcs is not None:
        if len(srcs) != len(outs):
            raise ValueError(f'Incorrect length for data "{outs.filename}" -- outputs: {len(outs)} vs. sources: {len(srcs)}')
        values['sources_file'] = srcs.filename
        for metric_class in metrics_dict['sourced_metrics']:
            metric = metric_class()
            values.update(metric.compute(outs, srcs))

    # compute ref-src-based metrics
    if refs is not None and srcs is not None:
        if len(srcs) != len(outs):
            raise ValueError(f'Incorrect length for data "{outs.filename}" -- outputs: {len(outs)} vs. sources: {len(srcs)}')
        values['references_file'] = refs.filename
        for metric_class in metrics_dict['sourced_and_referenced_metrics']:
            metric = metric_class()
            values.update(metric.compute(outs, refs, srcs))

    return values


def process_submission(outs: Submission, refs: Optional[Dict], srcs: Optional[Dict], metrics_dict: Dict[str, List]) -> Dict:
    """Process a (potentially) multi-dataset submission. Expects a Submission object
    holding all the predictions, and potentially references and/or sources in a dictionary keyed by
    dataset name, paralleling the datasets in the submission.

    If no references/sources are given and the dataset names correspond to GEM task datasets,
    default references/sources are used.

    Returns a dict keyed by dataset names, containing the dicts for each dataset's results.
    """
    values = {'submission_name': outs.name,
              'param_count': outs.param_count}
    for dataset in outs.datasets:
        logger.info(f'Computing metrics for {dataset}...')
        outs_ds = outs.predictions_for(dataset)
        # use default reference files if no custom ones are provided
        refs_ds = refs[dataset] if (refs and dataset in refs) else load_references(dataset)
        srcs_ds = srcs[dataset] if (srcs and dataset in srcs) else load_sources(dataset)
        values[dataset] = compute(outs_ds, refs_ds, srcs_ds, metrics_dict=metrics_dict)
    return values


# URLs to download standard references from
_DATASET_REFERENCES_URLS = {
    'asset_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/asset_test.json',
    'asset_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/asset_val.json',
    'common_gen_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/common_gen_val.json',
    'cs_restaurants_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/cs_restaurants_test.json',
    'cs_restaurants_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/cs_restaurants_val.json',
    'dart_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/dart_test.json',
    'dart_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/dart_val.json',
    'e2e_nlg_cleaned_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/e2e_nlg_cleaned_test.json',
    'e2e_nlg_cleaned_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/e2e_nlg_cleaned_val.json',
    'mlsum_de_val': 'https://github.com/GEM-bdechmark/GEM-metrics/releases/download/data/mlsum_de_val.json',
    'mlsum_de_test': 'https://github.com/GEM-bdechmark/GEM-metrics/releases/download/data/mlsum_de_test.json',
    'mlsum_es_val': 'https://github.com/GEM-beschmark/GEM-metrics/releases/download/data/mlsum_es_val.json',
    'mlsum_es_test': 'https://github.com/GEM-beschmark/GEM-metrics/releases/download/data/mlsum_es_test.json',
    'schema_guided_dstc8_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/schema_guided_dstc8_test.json',
    'schema_guided_dstc8_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/schema_guided_dstc8_val.json',
    'totto_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/totto_val.json',
    'turk_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/turk_test.json',
    'turk_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/turk_val.json',
    'webnlg_en_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/webnlg_en_val.json',
    'webnlg_en_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/webnlg_en_test.json',
    'webnlg_ru_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/webnlg_ru_val.json',
    'webnlg_ru_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/webnlg_ru_test.json',
    'wiki_lingua_es_en_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/wiki_lingua_es_en_test.json',
    'wiki_lingua_es_en_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/wiki_lingua_es_en_val.json',
    'wiki_lingua_ru_en_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/wiki_lingua_ru_en_test.json',
    'wiki_lingua_ru_en_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/wiki_lingua_ru_en_val.json',
    'wiki_lingua_tr_en_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/wiki_lingua_tr_en_test.json',
    'wiki_lingua_tr_en_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/wiki_lingua_tr_en_val.json',
    'wiki_lingua_vi_en_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/wiki_lingua_vi_en_test.json',
    'wiki_lingua_vi_en_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/wiki_lingua_vi_en_val.json',
    'xsum_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/xsum_test.json',
    'xsum_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/xsum_val.json',
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


_DATASET_SOURCES_URLS = {
    'asset_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/asset_test.json',
    'asset_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/asset_val.json',
    'turk_test': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/turk_test.json',
    'turk_val': 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/turk_val.json',
}


def load_sources(dataset_name: str) -> Optional[References]:
    """Load a file with sources for a standard GEM dataset (attempt download), return None if not present.
    Note that it can be the same files as for references -- the difference is in the source/target fields inside the JSON structure."""
    if dataset_name in _DATASET_SOURCES_URLS:
        try:
            dataset_file = ensure_download('references', dataset_name + '.json', _DATASET_SOURCES_URLS[dataset_name])
            return Sources(dataset_file)
        except Exception as e:
            logger.warn(f'Could not download references for {dataset_name}: {str(e)}')
            return None
    return None
