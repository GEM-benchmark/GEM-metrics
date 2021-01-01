#!/usr/bin/env python3

from nltk.translate import meteor_score
from .metric import ReferencedMetric
from .nltk_data import nltk_ensure_download
import numpy as np


class Meteor(ReferencedMetric):
    """METEOR uses NLTK implementation. This probably doesn't have the same results as the official
    METEOR implementation, but at least it doesn't need Java to work."""

    def __init__(self):
        nltk_ensure_download('corpora/wordnet')

    def compute(self, predictions, references):

        scores = []
        for refs, pred in zip(references.whitespace_tokenized, predictions.whitespace_tokenized):
            scores.append(meteor_score.meteor_score(refs, pred))
        return {'meteor': np.mean(scores)}
