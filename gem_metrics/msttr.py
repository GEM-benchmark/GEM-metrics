#!/usr/bin/env python3
import itertools

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

    def support_caching(self):
        # MSTTR is corpus-level, so individual examples can't be aggregated.
        return False

    def compute(self, cache, predictions):
        return {
            f"msttr-{self.window_size}": round(
                self._MSTTR(predictions.list_tokenized_lower, self.window_size)[
                    "msttr_value"
                ],
                5,
            ),
            f"msttr-{self.window_size}_nopunct": round(
                self._MSTTR(predictions.list_tokenized_lower_nopunct, self.window_size)[
                    "msttr_value"
                ],
                5,
            ),
        }

    def _TTR(self, list_of_words):
        "Compute type-token ratio."
        return len(set(list_of_words)) / len(list_of_words)

    def _MSTTR(self, tokenized_data, window_size):
        """
        Computes Mean-Segmental Type-Token Ratio (MSTTR; Johnson, 1944)
        by dividing the concatenated texts into non-overlapping segments of equal
        size and then averaging the TTRs of the segments.
        The last segment is excluded from the computation if it is smaller than
        the window size.
        """
        ttrs = []
        concatenated = list(itertools.chain.from_iterable(tokenized_data))
        
        for i in range(0, len(concatenated), window_size):
            window = concatenated[i: i+window_size]
            # removes the last segment from the computation
            if len(window) < window_size:
                break
            ttrs.append(self._TTR(window))

        results = {
            "msttr_value": sum(ttrs) / len(ttrs) if ttrs else float("nan"),
            "num_ttrs": len(ttrs),
            "ttrs": ttrs,
        }
        return results

    def _repeated_MSTTR(self, tokenized_data, window_size, repeats=5):
        "Repeated MSTTR to obtain a more robust MSTTR value."
        msttrs = []
        for i in range(repeats):
            sentences = self.rnd.sample(tokenized_data, len(tokenized_data))
            msttr_results = self._MSTTR(sentences, window_size)
            msttrs.append(msttr_results["msttr_value"])
        results = sum(msttrs) / len(msttrs)
        return results
