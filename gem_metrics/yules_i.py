#!/usr/bin/env python3

import itertools

from .metric import ReferencelessMetric
from .texts import Predictions

from typing import Dict


class Yules_I(ReferencelessMetric):
    """Yules I measure (Yules_I) as described in
    "Machine Translationese: Effects of Algorithmic Bias on Linguistic
    Complexity in Machine Translation", Vanmassenhove et al., EACL 2021.

    Yule’s characteristic constant (Yule’s K) measures constancy of text
    as the repetitiveness of vocabulary. The larger the Yule's K,
    the less rich the vocabulary is. Yule’s K and its inverse Yule’s I
    (implemented here) are considered to be more resilient to fluctuations
    related to text length than TTR.
    For a history of constancy measures of text, see Tanaka-Ishii et al,
    "Computational Constancy Measures of Texts—Yule's K and Rényi's Entropy".
    The implementation follows equation (1) in the above paper.
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

        """Computing Yules I measure
        :param sentences: dictionary with all words and their frequencies
        :returns: Yules I (the inverse of yule's K measure) (float) - the higher the better
        """
        _, vocabulary = self.get_vocabulary(predictions.untokenized)

        M1 = float(len(vocabulary))
        M2 = sum(
            [
                len(list(g)) * (freq**2)
                for freq, g in itertools.groupby(sorted(vocabulary.values()))
            ]
        )

        if M2 - M1 == 0:
            score = 0.0
        else:
            score = (M1 * M1) / (M2 - M1)

        return {"yules_i": round(score, 3)}
