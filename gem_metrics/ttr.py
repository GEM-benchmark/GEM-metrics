#!/usr/bin/env python3

from numpy import NaN
from .metric import ReferencelessMetric
from .texts import Predictions

from typing import Dict


class TTR(ReferencelessMetric):
    """Implements Type to Token Ratio (TTR) as described in 
    "Machine Translationese: Effects of Algorithmic Bias on Linguistic
    Complexity in Machine Translation", Vanmassenhove et al., EACL 2021.

    TTR presents the ratio of the total number of different words (types)
    to the total number of words (tokens). 
    Higher TTR indicates a higher degree of lexical diversity.
    This is implemented below using a simple dictionary counter.    
    """

    def support_caching(self):
        # corpus-level, so individual examples can't be aggregated.
        return False

    def get_vocabulary(self, sentence_array):
        """Compute vocabulary
        :param sentence_array: a list of sentences
        :returns: a list of tokens
        """
        data_vocabulary = {}
        total = 0
        
        for sentence in sentence_array:
            for token in sentence.strip().split():
                if token not in data_vocabulary:
                    data_vocabulary[token] = 1
                else:
                    data_vocabulary[token] += 1
                total += 1
                
        return total, data_vocabulary

    def compute(self, cache, predictions: Predictions) -> Dict:
        total, vocabulary = self.get_vocabulary(predictions.untokenized)
        if total == 0:
            score = NaN
        else:
            score = len(vocabulary)/total
        return {"ttr": round(score, 5)}