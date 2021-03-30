from nubia_score import Nubia
import numpy as np
from .metric import ReferencedMetric


class NUBIA(ReferencedMetric):
    def __init__(self):
        """Downloads pretrained models for nubia, and loads them into memory"""
        self.metric = Nubia()

    def compute(self, predictions, references):
        """Run Nubia"""
        scores = []
        for ref, pred in zip(references.untokenized, predictions.untokenized):
            scores.append(self.metric.score(ref, pred))
        return {"nubia": np.mean(scores)}
