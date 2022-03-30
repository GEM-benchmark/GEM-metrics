#!/usr/bin/env python3

from collections import defaultdict
from typing import Dict, List

import numpy as np
from moverscore_v2 import word_mover_score

from .metric import ReferencedMetric
from .texts import Predictions, References


class MoverScore(ReferencedMetric):
    """MoverScore uses the original implementation at https://github.com/AIPHES/emnlp19-moverscore"""

    def support_caching(self):
        return False

    def compute(self, cache, predictions: Predictions, references: References) -> Dict:

        moverscore = self.compute_score(predictions.untokenized, references.untokenized)
        return {"moverscore": round(moverscore, 5)}

    @staticmethod
    def sentence_score(hypothesis: str, references: List[str], trace=0):
        idf_dict_hyp = defaultdict(lambda: 1.0)
        idf_dict_ref = defaultdict(lambda: 1.0)

        hypothesis = [hypothesis] * len(references)

        scores = word_mover_score(
            references,
            hypothesis,
            idf_dict_ref,
            idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=False,
        )

        sentence_score = np.mean(scores)

        if trace > 0:
            print(hypothesis, references, sentence_score)

        return sentence_score

    def compute_score(self, hyp, refs):
        corpus_score = 0
        for i, j in enumerate(hyp):
            corpus_score += self.sentence_score(j, refs[i])
        return corpus_score / len(hyp)
