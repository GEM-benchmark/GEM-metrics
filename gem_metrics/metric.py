#!/usr/bin/env python3
from .texts import Predictions, References, Sources

from copy import copy
import numpy as np
from typing import List, Dict
from logzero import logger


class AbstractMetric:
    def compute(self):
        raise NotImplementedError

    def support_caching(self):
        return True

    def _initialize(self):
        """Function that initializes heavy models outside of the __init___."""
        pass

    def _aggregate_scores(self, score_list: List):
        """Helper function to aggregate multiple scores into a single one."""
        if not score_list:
            return {}
        if isinstance(score_list[0], float):
            raise ValueError(
                "A cached metric should return a dictionary for each datapoint."
            )
        elif isinstance(score_list[0], dict):
            l1_keys = list(score_list[0].keys())
            if isinstance(list(score_list[0].values())[0], float):
                return {
                    key: round(np.mean([score[key] for score in score_list]), 5)
                    for key in l1_keys
                }
            elif isinstance(list(score_list[0].values())[0], dict):
                l2_keys = score_list[0][l1_keys[0]].keys()
                return {
                    key1: {
                        key2: round(
                            np.mean([score[key1][key2] for score in score_list]), 5
                        )
                        for key2 in l2_keys
                    }
                    for key1 in l1_keys
                }
        else:
            return ValueError(
                "Please add to this function an aggregator for your data format."
            )

    def compute_cached(self, cache, predictions: Predictions, *args):
        """Loops through the predictions to check for cache hits before computing."""
        original_order = copy(predictions.ids)

        to_compute = []
        cached_scores = {}
        # Loop over IDs to check what needs to be computed and what is cached.
        if cache is not None and self.support_caching():
            for pred_id in predictions.ids:
                cache_key = (self.__class__.__name__, predictions.filename, pred_id)
                current_score = cache.get(cache_key, None)

                if current_score is not None:
                    cached_scores[pred_id] = current_score
                else:
                    to_compute.append(pred_id)
        else:
            to_compute = predictions.ids

        # Compute the rest if anything is left to compute.
        computed_scores = {}
        if to_compute:
            # Initialize in case it is defined (heavy metrics).
            self._initialize()
            # Each class needs filter() to list of ID in order.
            # Filtering done on copy to avoid destroying the underlying obj.
            new_arg_list = []
            for arg in [predictions] + list(args):
                new_arg = copy(arg)
                new_arg.assign_ids_and_unscramble(to_compute)
                new_arg_list.append(new_arg)
            computed_scores = self.compute(cache, *new_arg_list)
        else:
            logger.info(
                f"Everything in {self.__class__.__name__} for {predictions.filename} was cached :)"
            )

        if self.support_caching():
            # Combine them back and reshuffle.
            cached_scores.update(computed_scores)
            scores = cached_scores
            scores_ordered = [scores[pred_id] for pred_id in original_order]
            # Aggregate individual scores.
            aggregated_score = self._aggregate_scores(scores_ordered)
        else:
            # If module does not support caching, just return module output directly.
            aggregated_score = computed_scores
        return aggregated_score


class ReferencelessMetric(AbstractMetric):
    """Base class for all referenceless metrics."""

    def compute(self, cache, predictions: Predictions) -> Dict:
        raise NotImplementedError


class ReferencedMetric(AbstractMetric):
    """Base class for all referenced metrics."""

    def compute(self, cache, predictions: Predictions, references: References) -> Dict:
        raise NotImplementedError


class SourceAndReferencedMetric(AbstractMetric):
    """Base class for all metrics that require source and reference sentences."""

    def compute(self, cache, predictions: Predictions, references: References, sources: Sources) -> Dict:
        raise NotImplementedError
