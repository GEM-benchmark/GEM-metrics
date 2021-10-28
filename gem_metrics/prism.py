#!/usr/bin/env python3

from repro.models.thompson2020 import Prism as _Prism

from .metric import ReferencedMetric


class Prism(ReferencedMetric):
    def __init__(self, device: int, language: str = "en"):
        self.prism = _Prism(device=device, language=language)

    def compute(self, cache, predictions, references):
        inputs = []
        for pred, refs in zip(predictions.untokenized, references.untokenized):
            inputs.append({
                "candidate": pred,
                "references": refs
            })

        # Example `micro` output for 3 inputs
        # [{'prism': -1.1578280925750732}, {'prism': -1.3325390815734863}, {'prism': -2.730839729309082}]
        _, micro = self.prism.predict_batch(inputs)

        # Write to cache if not None and collect outputs
        id_to_scores = {}
        for pred_id, score_dict in zip(predictions.ids, micro):
            id_to_scores[pred_id] = score_dict
            if cache is not None:
                cache_key = (self.__class__.__name__, predictions.filename, pred_id)
                cache[cache_key] = score_dict

        return id_to_scores