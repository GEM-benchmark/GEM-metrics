import numpy as np
from typing import Dict

from .metric import ReferencedMetric
from .texts import Predictions, References


class WER(ReferencedMetric):
    """Word Error Rate is a similarity error measure, executed on lower case data without punctuation

    This implementation is based on scripts from Josef Hölzl et al. at:
        https://github.com/evanmiltenburg/NLG-diversity/blob/main/diversity.py
    Lower is better. The score range is [0,∞).
    """

    def support_caching(self):
        return False

    @staticmethod
    def get_wer(r, h):
        """
        This function is to calculate the edit distance of reference sentence and the hypothesis sentence.
        Main algorithm used is dynamic programming.
        Attributes:
            r -> the list of words produced by splitting reference sentence.
            h -> the list of words produced by splitting hypothesis sentence.
        """
        d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape(
            (len(r) + 1, len(h) + 1)
        )
        for i in range(len(r) + 1):
            d[i][0] = i
        for j in range(len(h) + 1):
            d[0][j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitute = d[i - 1][j - 1] + 1
                    insert = d[i][j - 1] + 1
                    delete = d[i - 1][j] + 1
                    d[i][j] = min(substitute, insert, delete)
        result = float(d[len(r)][len(h)]) / len(r) * 100
        return result

    def compute_score(self, prediction, references):
        scores = []
        for ref in references:
            scores.append(self.get_wer(ref, prediction))
        return np.min(scores)

    def compute(self, cache, predictions: Predictions, references: References) -> Dict:
        refs = references.list_tokenized_lower_nopunct
        preds = predictions.list_tokenized_lower_nopunct
        scores = [self.compute_score(i, j) for i, j in zip(preds, refs)]
        return {"wer": round(np.mean(scores), 5)}
