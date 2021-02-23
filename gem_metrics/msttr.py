#!/usr/bin/env python3

from string import punctuation
from nltk import ngrams
import random

from .metric import ReferencelessMetric


class MSTTR(ReferencelessMetric):
    """Mean segmental type-token ratio (based on tokenized data). Segment length is
    pre-set to 100 by default, computation is done on lowercased data. Returns two variants -- with
    and without taking punctuation into account.

    This is based on Emiel van Miltenburg's scripts from:
    https://github.com/evanmiltenburg/NLG-diversity/blob/main/diversity.py
    """
    def __init__(self, window_size=100):
        # use MSTTR-100 by default.
        self.rnd = random.Random(1234)
        self.window_size = window_size


    def compute(self, predictions):

        return {f'msttr-{self.window_size}': self._MSTTR(predictions.list_tokenized_lower, self.window_size)['msttr_value'],
                f'msttr-{self.window_size}_nopunct': self._MSTTR(predictions.list_tokenized_lower_nopunct, self.window_size)['msttr_value']}

    def _TTR(self, list_of_words):
        "Compute type-token ratio."
        return len(set(list_of_words)) / len(list_of_words)

    def _MSTTR(self, tokenized_data, window_size):
        "Compute Mean-Segmental Type-Token Ratio (MSTTR; Johnson, 1944)."
        chunk = []
        ttrs = []
        for sentence in tokenized_data:
            chunk_length = len(chunk)
            sentence_length = len(sentence)
            combined = chunk_length + sentence_length
            if combined < window_size:
                chunk.extend(sentence)
            elif combined == window_size:
                chunk.extend(sentence)
                ttrs.append(self._TTR(chunk))
                chunk = []
            else:
                needed = window_size - chunk_length
                chunk.extend(sentence[:needed])
                ttrs.append(self._TTR(chunk))
                chunk = sentence[needed:]
        results = {'msttr_value': sum(ttrs) / len(ttrs) if ttrs else float('nan'),
                   'num_ttrs': len(ttrs),
                   'ttrs': ttrs}
        return results

    def _repeated_MSTTR(self, tokenized_data, window_size, repeats=5):
        "Repeated MSTTR to obtain a more robust MSTTR value."
        msttrs = []
        for i in range(repeats):
            sentences = self.rnd.sample(tokenized_data, len(tokenized_data))
            msttr_results = self._MSTTR(sentences, window_size)
            msttrs.append(msttr_results['msttr_value'])
        results = sum(msttrs) / len(msttrs)
        return results


