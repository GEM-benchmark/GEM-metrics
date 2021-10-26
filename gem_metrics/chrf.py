#!/usr/bin/env python3

from .metric import ReferencedMetric

from itertools import zip_longest

import sacrebleu

class CHRF(ReferencedMetric):
    """
    Computes CHRF, CHRF+ and CHRF++.

    CHRF is a language-agnostic metric introduced in:
   
    CHRF: character n-gram F-score for automatic MT evaluation
    Maja Popovic 
    Proceedings of the Tenth Workshop on Statistical Machine Translation, ACL, 2015
    https://aclanthology.org/W15-3049.pdf
   
    CHRF calculates an F-score based on character n-gram precision (CHRP) and 
    character n-gram recall (CHRR) as:
    CHRF = (1+beta^2) * ((CHRP * CHRR) / (beta^2 * (CHRP + CHRR))
    This implementation wraps sacrebleu (https://github.com/mjpost/sacrebleu) and uses
    beta = 2 as well as epsilon smoothing similar to the reference implementation.
    Character n-grams up to the order of 6 are considered.


    CHRF++ was introduced in:

    CHRF++: words helping character n-grams
    Maja Popovic
    Proceedings of the Conference on Machine Translation (WMT), ACL, 2017
    https://aclanthology.org/W17-4770.pdf

    and adds word unigrams and bigrams to the metric.
    In CHRF+, only unigrams are added.
    """

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
                word_order=word_order,
                eps_smoothing=True
            )
            scores[key] = chrf.score

        return scores
