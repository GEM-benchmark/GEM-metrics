#!/usr/bin/env python3

from .metric import ReferencedMetric

from itertools import zip_longest

import sacrebleu

class CHRF(ReferencedMetric):
    """CHRF from SacreBLEU."""

    def support_caching(self):
        # corpus-level, so individual examples won't be aggregated.
        return False

    def compute(self, cache, predictions, references):
        ref_streams = list(zip_longest(*references.untokenized))
        scores = {}
        
        for word_order in range(0, 3):
            key = "chrf" + "+" * word_order
            chrf = sacrebleu.corpus_chrf(
                predictions.untokenized, 
                ref_streams,
                word_order=word_order
            )
            scores[key] = round(chrf.score, 5)

        return scores
