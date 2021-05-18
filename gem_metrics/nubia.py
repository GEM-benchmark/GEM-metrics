from nubia_score import Nubia
import numpy as np
from .metric import ReferencedMetric


class NUBIA(ReferencedMetric):
    def __init__(self):
        """Downloads pretrained models for nubia, and loads them into memory"""
        self.metric = Nubia()
        self.metric.roberta_STS.to("cuda")
        self.metric.roberta_MNLI.to("cuda")
        self.metric.gpt_model.to("cuda")

    def compute(self, cache, predictions, references):
        """Run Nubia"""
        scores = {}
        for ref, pred, pred_id in zip(
            references.untokenized, predictions.untokenized, predictions.ids
        ):
            if isinstance(ref, list):
                # For multi-reference data, compute micro-average.
                multi_scores = []
                for _ref in ref:
                    multi_scores.append(
                        self.metric.score(_ref, pred, get_features=True)
                    )

                features = {
                    key: np.mean([s["features"][key] for s in multi_scores])
                    for key in multi_scores[0]["features"].keys()
                }
                features["nubia_score"] = np.mean(
                    [s["nubia_score"] for s in multi_scores]
                )
                score = {"nubia": features}
            else:
                score = {"nubia": self.metric.score(ref, pred, get_features=True)}
            # Write to cache if not None.
            if cache is not None:
                cache_key = (self.__class__.__name__, predictions.filename, pred_id)
                cache[cache_key] = score
            scores[pred_id] = score
        return scores
