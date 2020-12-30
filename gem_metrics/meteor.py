#!/usr/bin/env python3

from nltk.translate import meteor_score
from .metric import Metric
from .nltk_data import nltk_ensure_download
import numpy as np


class Meteor(Metric):

    def __init__(self):
        nltk_ensure_download('corpora/wordnet')

    def compute(self, predictions, references):

        scores = []
        for refs, pred in zip(references.whitespace_tokenized, predictions.whitespace_tokenized):
            scores.append(meteor_score.meteor_score(refs, pred))
        return {'meteor': np.mean(scores)}
