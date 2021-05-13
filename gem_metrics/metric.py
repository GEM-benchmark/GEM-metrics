#!/usr/bin/env python3
from copy import copy
import numpy as np
from typing import List

class AbstractMetric:
  def compute(self):
    raise NotImplementedError


  def _aggregate_scores(self, score_list: List):
    if isinstance(score_list[0], float):
      return np.mean(score_list)
    elif isinstance(score_list[0], dict):
      l1_keys = list(score_list[0].keys())
      if isinstance(list(score_list[0].values())[0], float):
        return {key: np.mean([score[key] for score in score_list]) for key in l1_keys}
      elif isinstance(list(score_list[0].values())[0], dict):
        l2_keys = score_list[0][l1_keys[0]].keys()
        return {key1: {key2: np.mean([score[key1][key2] for score in score_list]) for key2 in l2_keys} for key1 in l1_keys}
    else:
      return ValueError("Please add to this function an aggregator for your data format.")

  def compute_cached(self, cache, predictions, *args):
    """Loops through the predictions to check for cache hits before computing."""
    original_order = copy(predictions.ids)
    to_compute = []
    scores = {}
    # Loop over IDs to check what needs to be computed.
    if cache is not None:
      for pred_id in predictions.ids:
        cache_key = (self.__class__.__name__, predictions.filename, pred_id)
        current_score = cache.get(cache_key, None)
        if current_score is not None:
          scores[pred_id] = current_score
        else:
          to_compute.append(pred_id)
    else:
      to_compute = predictions.ids

    # Compute the rest if anything is left to compute.
    additional_scores = {}
    if to_compute:
      # Each class needs filter() to list of ID in order.
      # Filtering done on copy to avoid destroying the underlying obj. 
      new_arg_list = []
      for arg in [predictions] + list(args):
        new_arg = copy(arg)
        new_arg.assign_ids_and_unscramble(to_compute)
        new_arg_list.append(new_arg)
      additional_scores = self.compute(cache, *new_arg_list)

    # Combine them back and reshuffle.
    scores = scores | additional_scores
    scores_ordered = [scores[pred_id] for pred_id in original_order]

    return self._aggregate_scores(scores_ordered)


class ReferencelessMetric(AbstractMetric):
  """Base class for all referenceless metrics."""

  def compute(self, cache, predictions):
    raise NotImplementedError


class ReferencedMetric(AbstractMetric):
  """Base class for all referenced metrics."""

  def compute(self, cache, predictions, references):
    raise NotImplementedError


class SourceAndReferencedMetric(AbstractMetric):
  """Base class for all metrics that require source and reference sentences."""

  def compute(self, cache, predictions, references, sources):
    raise NotImplementedError
