#!/usr/bin/env python3

from .metric import ReferencedMetric
import sacrebleu
from itertools import zip_longest


class BLEU(ReferencedMetric):
    """BLEU uncased BLEU from SacreBLEU."""

    def support_caching(self):
        # BLEU is corpus-level, so individual examples can't be aggregated.
        return False

    def compute(self, cache, predictions, references):
        ref_streams = list(zip_longest(*references.untokenized))
        bleu = sacrebleu.corpus_bleu(
            predictions.untokenized, ref_streams, lowercase=True
        )
        return {"bleu": round(bleu.score, 5)}
