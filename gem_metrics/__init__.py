#!/usr/bin/env python3

from argparse import ArgumentParser
from copy import copy
from dataclasses import dataclass
from diskcache import Cache
import json
from multiprocessing import Process, Manager
from multiprocessing.pool import ThreadPool as Pool

from typing import Optional, Dict, List
import sys
import traceback
from logzero import logger

# Data holder classes
from .texts import Predictions, References, Sources, Submission

# Metric implementations
from .meteor import Meteor
from .bertscore import BERTScore
from .bleu import BLEU
from .bleurt import BLEURT
from .rouge import ROUGE
from .nist import NIST
from .local_recall import LocalRecall
from .msttr import MSTTR
from .ngrams import NGramStats
from .data import ensure_download
from .sari import SARI
from .nubia import NUBIA

# from .questeval import QuestEval


def metric_list_to_metric_dict(metric_list: List[str]) -> Dict[str, List]:
    """
    Function that converts a list of strings corresponding to the metric names into a dictionary with three keys,
    referenced_metrics, referenceless_metrics, and sourced_and_referenced_metrics, which are populated by the actual metrics class.
    """
    # convert to set in case there are repeats
    metric_list = list(set(metric_list))

    metric_name_to_metric_class = {
        "bertscore": BERTScore,
        "bleu": BLEU,
        "bleurt": BLEURT,
        "local_recall": LocalRecall,
        "meteor": Meteor,
        "nist": NIST,
        "rouge": ROUGE,
        "msttr": MSTTR,
        "ngram": NGramStats,
        "sari": SARI,
        "nubia": NUBIA,
        # 'questeval': QuestEval,
    }

    metric_name_to_metric_type = {
        "bertscore": "referenced",
        "bleu": "referenced",
        "bleurt": "referenced",
        "local_recall": "referenced",
        "meteor": "referenced",
        "nist": "referenced",
        "rouge": "referenced",
        "msttr": "referenceless",
        "ngram": "referenceless",
        "sari": "sourced_and_referenced",
        "nubia": "referenced",
        # "questeval": "sourced_and_referenced",
    }

    referenced_list, referenceless_list, sourced_and_referenced_list = [], [], []

    for metric_name in metric_list:
        metric_class = metric_name_to_metric_class[metric_name]
        metric_type = metric_name_to_metric_type[metric_name]
        if metric_type == "referenced":
            referenced_list.append(metric_class)
        elif metric_type == "referenceless":
            referenceless_list.append(metric_class)
        elif metric_type == "sourced_and_referenced":
            sourced_and_referenced_list.append(metric_class)
        else:
            raise NotImplementedError(
                f"{metric_type} is not one of [referenced, referenceless, sourced_and_referenced]. Please check the metric_name_to_metric_type dict."
            )

    metric_dict = {
        "referenced_metrics": referenced_list,
        "referenceless_metrics": referenceless_list,
        "sourced_and_referenced_metrics": sourced_and_referenced_list,
    }

    return metric_dict


def compute(
    outs: Predictions,
    refs: Optional[References] = None,
    srcs: Optional[Sources] = None,
    metrics_dict: Dict[str, List] = None,
    cache: Optional[Cache] = None,
) -> Dict:
    """Main metrics computation routine for a single dataset.

    Args:
      outs: texts.Predictions object.
      refs: texts.References object (optional).
      srcs: texts.Sources object (optional).
      metrics_dict: is a dictionary with three keys:
      referenced_metrics, referenceless_metrics, and sourced_and_referenced_metrics.
      Each of those keys' values are a List of the specific metrics.
      cache: a diskcache.Cache object for fast lookups of redundant computations.

    Returns:
      values: A dict with the results with metric names as keys.
    """
    # initialize values storage.
    values = {"predictions_file": outs.filename, "N": len(outs)}

    # compute referenceless metrics.
    for metric_class in metrics_dict["referenceless_metrics"]:
        logger.info(f"Computing {metric_class.__name__} for {outs.filename}...")
        metric = metric_class()
        values.update(metric.compute_cached(cache, outs))
        # Explicit deletion due to memory leak when multiple models were instantiated.
        del metric

    # compute ref-based metrics
    if refs is not None:
        if len(refs) != len(outs):
            raise ValueError(
                f'Incorrect length for data "{outs.filename}" -- outputs: {len(outs)} vs. references: {len(refs)}'
            )
        values["references_file"] = refs.filename
        for metric_class in metrics_dict["referenced_metrics"]:
            logger.info(f"Computing {metric_class.__name__} for {outs.filename}...")
            metric = metric_class()
            values.update(metric.compute_cached(cache, outs, refs))
            del metric

    # compute ref-src-based metrics
    if refs is not None and srcs is not None:
        if len(srcs) != len(outs):
            raise ValueError(
                f'Incorrect length for data "{outs.filename}" -- outputs: {len(outs)} vs. sources: {len(srcs)}'
            )
        values["references_file"] = refs.filename
        for metric_class in metrics_dict["sourced_and_referenced_metrics"]:
            logger.info(f"Computing {metric_class.__name__}...")
            metric = metric_class()
            values.update(metric.compute_cached(cache, outs, refs, srcs))
            del metric
    return values


def process_submission(
    outs: Submission,
    refs: Optional[Dict],
    srcs: Optional[Dict],
    parallel_metric_dict: Dict[str, List],
    serial_metric_dict: Dict[str, List],
    cache: Optional[Cache] = None,
) -> Dict:
    """Process a (potentially) multi-dataset submission. Expects a Submission object
    holding all the predictions, and potentially references and/or sources in a dictionary keyed by
    dataset name, paralleling the datasets in the submission.

    If no references/sources are given and the dataset names correspond to GEM task datasets,
    default references/sources are used.

    Returns a dict keyed by dataset names, containing the dicts for each dataset's results.
    """
    # Handle the CPU-bound metrics in parallel to speed up computation.
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict["submission_name"] = outs.name
    shared_dict["param_count"] = outs.param_count

    def multiprocess_compute(dataset, outs_ds, refs_ds, srcs_ds, metrics_dict, cache):
        shared_dict[dataset] = compute(outs_ds, refs_ds, srcs_ds, metrics_dict, cache)

    job_args = []

    for dataset in outs.datasets:
        logger.info(f"Computing metrics for {dataset}...")
        outs_ds = outs.predictions_for(dataset)
        refs_ds = refs.get(dataset, None)
        srcs_ds = srcs.get(dataset, None)
        job_args.append(
            (dataset, outs_ds, refs_ds, srcs_ds, parallel_metric_dict, cache)
        )

    pool = Pool(processes=len(job_args))
    pool.starmap(multiprocess_compute, [x for x in job_args])
    pool.close()
    pool.join()

    logger.info("Moving on to the serial metrics now.")

    for dataset in outs.datasets:
        logger.info(f"Computing serial metrics for {dataset}...")
        outs_ds = outs.predictions_for(dataset)
        refs_ds = refs.get(dataset, None)
        srcs_ds = srcs.get(dataset, None)
        shared_dict[dataset] = shared_dict[dataset] | compute(
            outs_ds, refs_ds, srcs_ds, serial_metric_dict, cache
        )

    return dict(shared_dict)


# URLs to download standard references from
_SUPPORTED_DATASETS = {
    "cs_restaurants_val": "cs",
    "cs_restaurants_test": "cs",
    "cs_restaurants_challenge_test_scramble": "cs",
    "common_gen_val": "cs",
    # "common_gen_test": "en",
    # "common_gen_challenge_test_scramble": "en",
    "dart_val": "en",
    "dart_test": "en",
    "e2e_nlg_val": "en",
    "e2e_nlg_test": "en",
    "e2e_nlg_challenge_test_scramble": "en",
    "mlsum_de_val": "de",
    "mlsum_de_test": "de",
    "mlsum_de_challenge_test_covid": "de",
    "mlsum_es_val": "es",
    "mlsum_es_test": "es",
    "mlsum_es_challenge_test_covid": "es",
    "schema_guided_dialog_val": "en",
    "schema_guided_dialog_test": "en",
    "schema_guided_dialog_challenge_test_backtranslation": "en",
    "schema_guided_dialog_challenge_test_bfp02": "en",
    "schema_guided_dialog_challenge_test_bfp05": "en",
    "schema_guided_dialog_challenge_test_nopunc": "en",
    "schema_guided_dialog_challenge_test_scramble": "en",
    "totto_val": "en",
    # "totto_test": "en",
    # "totto_challenge_test_scramble": "en",
    "web_nlg_en_val": "en",
    "web_nlg_en_test": "en",
    "web_nlg_en_challenge_test_scramble": "en",
    "web_nlg_en_challenge_test_numbers": "en",
    "web_nlg_ru_val": "ru",
    "web_nlg_ru_test": "ru",
    "web_nlg_ru_challenge_test_scramble": "ru",
    "wiki_auto_asset_turk_val": "en",
    "wiki_auto_asset_turk_test_asset": "en",
    "wiki_auto_asset_turk_test_turk": "en",
    "wiki_auto_asset_turk_challenge_test_asset_backtranslation": "en",
    "wiki_auto_asset_turk_challenge_test_asset_bfp02": "en",
    "wiki_auto_asset_turk_challenge_test_asset_bfp05": "en",
    "wiki_auto_asset_turk_challenge_test_asset_nopunc": "en",
    "wiki_auto_asset_turk_challenge_test_turk_backtranslation": "en",
    "wiki_auto_asset_turk_challenge_test_turk_bfp02": "en",
    "wiki_auto_asset_turk_challenge_test_turk_bfp05": "en",
    "wiki_auto_asset_turk_challenge_test_turk_nopunc": "en",
    "wiki_lingua_spanish_es_val": "es",
    "wiki_lingua_spanish_es_test": "es",
    "wiki_lingua_russian_ru_val": "ru",
    "wiki_lingua_russian_ru_test": "ru",
    "wiki_lingua_turkish_tr_val": "tr",
    "wiki_lingua_turkish_tr_test": "tr",
    "wiki_lingua_vietnamese_vi_val": "vi",
    "wiki_lingua_vietnamese_vi_test": "vi",
    "xsum_val": "en",
    "xsum_test": "en",
    "xsum_challenge_test_backtranslation": "en",
    "xsum_challenge_test_bfp_02": "en",
    "xsum_challenge_test_bfp_05": "en",
    "xsum_challenge_test_nopunc": "en",
    "xsum_challenge_test_covid": "en",
}

# URLs to download standard references from
_CHALLENGE_SET_MATCHES = {
    "cs_restaurants_challenge_test_scramble": "cs_restaurants_test",
    "web_nlg_ru_challenge_test_scramble": "web_nlg_ru_test",
    "schema_guided_dialog_challenge_test_backtranslation": "schema_guided_dialog_test",
    "schema_guided_dialog_challenge_test_bfp02": "schema_guided_dialog_test",
    "schema_guided_dialog_challenge_test_bfp05": "schema_guided_dialog_test",
    "schema_guided_dialog_challenge_test_nopunc": "schema_guided_dialog_test",
    "schema_guided_dialog_challenge_test_scramble": "schema_guided_dialog_test",
    "xsum_challenge_test_backtranslation": "xsum_test",
    "xsum_challenge_test_bfp_02": "xsum_test",
    "xsum_challenge_test_bfp_05": "xsum_test",
    "xsum_challenge_test_nopunc": "xsum_test",
    "e2e_nlg_challenge_test_scramble": "e2e_nlg_test",
    "web_nlg_en_challenge_test_scramble": "web_nlg_en_test",
    "web_nlg_en_challenge_test_numbers": "web_nlg_en_test",
    "wiki_auto_asset_turk_challenge_test_asset_backtranslation": "wiki_auto_asset_turk_test_asset",
    "wiki_auto_asset_turk_challenge_test_asset_bfp02": "wiki_auto_asset_turk_test_asset",
    "wiki_auto_asset_turk_challenge_test_asset_bfp05": "wiki_auto_asset_turk_test_asset",
    "wiki_auto_asset_turk_challenge_test_asset_nopunc": "wiki_auto_asset_turk_test_asset",
    "wiki_auto_asset_turk_challenge_test_turk_backtranslation": "wiki_auto_asset_turk_test_turk",
    "wiki_auto_asset_turk_challenge_test_turk_bfp02": "wiki_auto_asset_turk_test_turk",
    "wiki_auto_asset_turk_challenge_test_turk_bfp05": "wiki_auto_asset_turk_test_turk",
    "wiki_auto_asset_turk_challenge_test_turk_nopunc": "wiki_auto_asset_turk_test_turk",
}

_CONTRAST_SET_BASE = [
    "cs_restaurants_test",
    "totto_test",
    "xsum_test",
    "web_nlg_en_test",
    "web_nlg_ru_test",
    "wiki_auto_asset_turk_test_asset",
    # "wiki_auto_asset_turk_test_turk",
]

_LANGUAGES = {}

_CONTRAST_SET_MATCHES = {
    dataset_name: f"https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/{dataset_name}_contrast_sets.json"
    for dataset_name in _CONTRAST_SET_BASE
}

_DATASET_REFERENCES_URLS = {
    dataset_name: f"https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/{dataset_name}.json"
    for dataset_name in _SUPPORTED_DATASETS
}
# Fix val -> validation in download.
for key, value in _DATASET_REFERENCES_URLS.items():
    if key.endswith("val"):
        _DATASET_REFERENCES_URLS[key] = value.replace("val", "validation")


def load_references(dataset_name: str) -> Optional[References]:
    """Load a file with references for a standard GEM dataset (attempt download), return None if not present."""
    if dataset_name in _DATASET_REFERENCES_URLS:
        try:
            dataset_file = ensure_download(
                "references",
                dataset_name + ".json",
                _DATASET_REFERENCES_URLS[dataset_name],
            )
            return References(dataset_file, language=_SUPPORTED_DATASETS[dataset_name])
        except Exception as e:
            logger.warn(f"Could not format references for {dataset_name}: {str(e)}")
            traceback.print_tb(e.__traceback__)
            return None
    return None


def load_sources(dataset_name: str) -> Optional[References]:
    """Load a file with references for a standard GEM dataset (attempt download), return None if not present."""
    if dataset_name in _DATASET_REFERENCES_URLS:
        try:
            dataset_file = ensure_download(
                "references",
                dataset_name + ".json",
                _DATASET_REFERENCES_URLS[dataset_name],
            )
            return Sources(dataset_file, language=_SUPPORTED_DATASETS[dataset_name])
        except Exception as e:
            logger.info(f"{dataset_name} does not have source associated.")
            return None
    return None


def load_contrast_set(dataset_name: str) -> Optional[Dict]:
    if dataset_name in _CONTRAST_SET_MATCHES:
        try:
            dataset_file = ensure_download(
                "contrast_sets",
                dataset_name + "_contrast_sets.json",
                _CONTRAST_SET_MATCHES[dataset_name],
            )
            with open(dataset_file) as f:
                contrast_sets = json.load(f)
            return contrast_sets
        except Exception as e:
            logger.warn(f"Could not format contrast set for {dataset_name}: {str(e)}")
            logger.warn(f"Looked for this file: {_CONTRAST_SET_MATCHES[dataset_name]}.")
            traceback.print_tb(e.__traceback__)
            return None
    return None


@dataclass
class Config:
    predictions_file: str = ""
    references_file: str = ""
    sources_file: str = ""
    output_file: str = ""
    use_heavy_metrics: bool = False
    metric_list: list = None
    cache_folder: str = ""


def process_files(config):
    """Main entry point -- load inputs, call metrics measuring, print outputs"""
    parallel_metric_dict = metric_list_to_metric_dict(config.metric_list)
    parallel_metrics_list = []
    if config.use_heavy_metrics:
        parallel_metrics_list.append("bertscore")
        # parallel_metrics_list.append("bleurt")
        parallel_metrics_list.append("nubia")
        # parallel_metrics_list.append("questeval")
    serial_metric_dict = metric_list_to_metric_dict(parallel_metrics_list)

    # Optionally, set up cache.
    cache = None
    if config.cache_folder:
        cache = Cache(config.cache_folder)
        cache.stats(enable=True)

    # load system predictions
    with open(config.predictions_file, encoding="UTF-8") as fh:
        data = json.load(fh)

    # multi-file submissions
    if isinstance(data, dict) and "submission_name" in data:
        data = Submission(data, language_table=_SUPPORTED_DATASETS)

        ref_data = {}
        if config.references_file:
            with open(config.references_file, encoding="UTF-8") as fh:
                ref_data = json.load(fh)
                for dataset in ref_data.keys():
                    ref_data[dataset] = References(
                        ref_data[dataset], language=_SUPPORTED_DATASETS[dataset]
                    )

        src_data = {}
        if config.sources_file:
            with open(config.sources_file, encoding="UTF-8") as fh:
                src_data = json.load(fh)
                for dataset in src_data.keys():
                    src_data[dataset] = Sources(
                        src_data[dataset], language=_SUPPORTED_DATASETS[dataset]
                    )

        # Use default reference+source files if no custom ones are provided.
        for dataset in data.datasets:
            if dataset not in ref_data:
                ref_data[dataset] = load_references(dataset)

            if dataset not in src_data:
                src_data[dataset] = load_sources(dataset)

            # Ensure that the reference files are ordered the same way.
            if ref_data[dataset] is not None:
                # Only if reference have IDs.
                if hasattr(ref_data[dataset], "ids"):
                    outs_ds = data.predictions_for(dataset)
                    outs_ds.assign_ids_and_unscramble(id_list=ref_data[dataset].ids)

        # For challenge sets, assume that we have a gem_parent_id and construct
        # the corresponding test subset.
        for dataset in data.datasets:
            # Only works if we have references.
            if ref_data[dataset] is not None:
                # And if the references have the parent_id set.
                if ref_data[dataset].has_parent_ids:
                    if dataset not in _CHALLENGE_SET_MATCHES:
                        logger.info(
                            "Found parent ID in %s but no corresponding parent dataset"
                            % dataset
                        )
                        continue
                    parent_dataset_name = _CHALLENGE_SET_MATCHES[dataset]
                    new_dataset_name = f"{dataset}_parent"
                    logger.info("Adding new subset dataset %s" % new_dataset_name)
                    # Construct new references.
                    new_refs = copy(ref_data[parent_dataset_name])
                    new_refs.assign_ids_and_unscramble(ref_data[dataset].parent_ids)
                    ref_data[new_dataset_name] = new_refs
                    # Construct new predictions.
                    new_preds = copy(data.predictions_for(parent_dataset_name))
                    new_preds.assign_ids_and_unscramble(ref_data[dataset].parent_ids)
                    data.entries[new_dataset_name] = new_preds
                    logger.info("Dataset successfully added.")

        # Next construct the contrast sets. The files define a list of IDs we can
        # match on.
        for dataset in data.datasets:
            if dataset in _CONTRAST_SET_MATCHES:
                # Assemble dictionary of all the subsets.
                contrast_sets = load_contrast_set(dataset)
                for set_name, subsets in contrast_sets.items():
                    for subset_name, id_list in subsets.items():
                        new_dataset_name = (
                            f"{dataset}_contrast_{set_name}-{subset_name}"
                        )
                        logger.info("Adding new contrast dataset %s" % new_dataset_name)
                        # Optionally, construct new references.
                        if ref_data[dataset] is not None:
                            new_refs = copy(ref_data[dataset])
                            new_refs.assign_ids_and_unscramble(id_list)
                            ref_data[new_dataset_name] = new_refs
                        # Construct new predictions.
                        new_preds = copy(data.predictions_for(dataset))
                        new_preds.assign_ids_and_unscramble(id_list)
                        data.entries[new_dataset_name] = new_preds
                        logger.info("Dataset successfully added.")
        # Compute all the values.
        values = process_submission(
            outs=data,
            refs=ref_data,
            srcs=src_data,
            parallel_metric_dict=parallel_metric_dict,
            serial_metric_dict=serial_metric_dict,
            cache=cache,
        )

    # Single-file mode.
    else:
        outs = Predictions(data)
        srcs = {}
        refs = {}

        # load references, if available
        if config.references_file is not None:
            refs = References(config.references_file)
            assert len(refs) == len(outs)

            # Ensure that they are not scrambled.
            outs.assign_ids_and_unscramble(id_list=refs.ids)

        # load sources, if available
        if config.sources_file is not None:
            srcs = Sources(config.sources_file)
            assert len(srcs) == len(outs)
        # In single file mode this is automatically all serial.
        serial_metric_dict.update(parallel_metric_dict)
        values = compute(outs, refs, srcs, serial_metric_dict, cache)

    # print output
    out_fh = sys.stdout
    if config.output_file:
        out_fh = open(config.output_file, "w", encoding="UTF-8")
    print(json.dumps(values, ensure_ascii=False, indent=4), file=out_fh)


def main():
    ap = ArgumentParser(description="GEM automatic metrics script")
    ap.add_argument(
        "predictions_file", type=str, help="Path to system outputs JSON file"
    )
    ap.add_argument(
        "-r",
        "--references-file",
        "--references",
        "--refs",
        type=str,
        help="Path to references JSON file",
    )
    ap.add_argument(
        "-s",
        "--sources-file",
        "--sources",
        "--srcs",
        type=str,
        help="Path to sources JSON file",
    )
    ap.add_argument(
        "-o", "--output-file", type=str, help="Path to output file", default=""
    )
    ap.add_argument(
        "--heavy-metrics",
        "--heavy_metrics",
        action="store_true",
        help="Run heavyweight metrics (BERTScore, BLEURT, NUBIA and QuestEval)",
    )
    ap.add_argument(
        "--metric-list",
        nargs="+",
        default=[
            "bleu",
            "meteor",
            "rouge",
            "nist",
            "msttr",
            "ngram",
            "sari",
            "local_recall",
        ],
        help=(
            "Full metric list default is [bleu, meteor, rouge, nist, msttr, ngram, sari, local_recall]. "
            + "You can add bertscore, bleurt, nubia and questeval by manually adding them in the command "
            + "line argument here, or by using the --heavy-metrics flag"
        ),
    )
    ap.add_argument(
        "--cache_folder",
        type=str,
        default="",
        help=(
            "Optional Path to a cache folder."
            "This script is computing many different metrics across many different model outputs."
            "It can thus become *very* slow, especially when the `heavy_metrics`"
            "setting is enabled. Since challenge and constrast sets may rerun already"
            "completed evaluations, and you may want to rerun the script for some"
            "reason, we can set up a disk-persistent key-value storage."
            "If this argument is specified, it will point to the caching folder. "
        ),
    )
    args = ap.parse_args()

    # Workaround for metrics that use cmd flags - write all args to config.
    config = Config(
        predictions_file=args.predictions_file,
        references_file=args.references_file,
        sources_file=args.sources_file,
        output_file=args.output_file,
        use_heavy_metrics=args.heavy_metrics,
        metric_list=args.metric_list,
        cache_folder=args.cache_folder,
    )

    # hack to make BLEURT work -- it'll fail for anything in argv except the program name :-(
    sys.argv = sys.argv[:1]
    process_files(config)
