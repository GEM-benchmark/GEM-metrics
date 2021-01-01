#!/usr/bin/env python3

from .metric import ReferencelessMetric
from .impl.diversity import MSTTR as MSTTR_func
from .impl.diversity import PUNCTUATION


class MSTTR(ReferencelessMetric):
    """Mean segmental type-token ratio (based on tokenized data)"""

    def compute(self, predictions):

        # compute on lowercase data, variant -- w/o punctuation
        lc_preds = [[w.lower() for w in item] for item in predictions.list_tokenized]
        no_punct = [[w for w in item if w not in PUNCTUATION] for item in lc_preds]

        # use MSTTR-100
        return {'msttr-100': MSTTR_func(lc_preds, 100)['msttr_value'],
                'msttr-100_nopunct': MSTTR_func(no_punct, 100)['msttr_value']}
