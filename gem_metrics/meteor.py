#!/usr/bin/env python3

from nltk.translate import meteor_score
from .metric import Metric


class Meteor(Metric):

    def compute(self, predictions, references):
        score = meteor_score.single_meteor_score(references.whitespace_tokenized, predictions.whitespace_tokenized)
        return {'meteor': score}
