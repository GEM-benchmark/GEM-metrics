#!/usr/bin/env python3
from .metric import ReproReferencedMetric

from repro.models.sellam2020 import BLEURT as _BLEURT
from typing import List


class BLEURT(ReproReferencedMetric):
    """BLEURT uses the base checkpoint for efficient runtime."""

    def __init__(self, checkpoint_path="bleurt-base-128", device: int = 0, batch_size: int = 64):
        metric = _BLEURT(device=device, model=checkpoint_path, batch_size=batch_size)
        super().__init__(metric)

    def _postprocess(self, score_dicts: List) -> List:
        # The Repro version contains two scores, one with a mean over multiple
        # references per instance and one with a max. The original GEM implementation
        # only had mean, so we post-process to make the two implementations identical.
        return [
            {
                "bleurt": scores["bleurt"]["mean"]
            }
            for scores in score_dicts
        ]
