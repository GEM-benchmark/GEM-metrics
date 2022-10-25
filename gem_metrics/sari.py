#!/usr/bin/env python3
from .texts import Predictions, References, Sources
from .metric import SourceAndReferencedMetric

from typing import Dict, Tuple, List
from collections import Counter
import sacrebleu
import sacremoses


class SARI(SourceAndReferencedMetric):
    """SARI score for evaluating paraphrasing and other text generation models.
    The score is introduced in the following paper:
       Optimizing Statistical Machine Translation for Text Simplification
       Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen and Chris Callison-Burch
       In Transactions of the Association for Computational Linguistics (TACL) 2015
       http://cs.jhu.edu/~napoles/res/tacl2016-optimizing.pdf
    This implementation is adapted from Tensorflow's tensor2tensor implementation [3].
    It has two differences with the original GitHub [1] implementation:
      (1) Define 0/0=1 instead of 0 to give higher scores for predictions that match
        a target exactly.
      (2) Fix an alleged bug [2] in the keep score computation.
    [1] https://github.com/cocoxu/simplification/blob/master/SARI.py
      (commit 0210f15)
    [2] https://github.com/cocoxu/simplification/issues/6
    [3] https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/sari_hook.py
    """

    def compute(
        self, cache, predictions: Predictions, references: References, sources: Sources
    ) -> Dict:

        srcs = [self.normalize(sent) for sent in sources.untokenized]
        preds = [self.normalize(sent) for sent in predictions.untokenized]
        refs = [
            [self.normalize(sent) for sent in ref_sents]
            for ref_sents in references.untokenized
        ]

        sari_scores = {}
        for i in range(len(srcs)):
            score = {"sari": self.SARIsent(srcs[i], preds[i], refs[i]) * 100}
            sari_scores[predictions.ids[i]] = score
            # Write to cache if not None.
            if cache is not None:
                cache_key = (
                    self.__class__.__name__,
                    predictions.filename,
                    predictions.ids[i],
                )
                cache[cache_key] = score

        return sari_scores

    def SARIngram(
        self,
        sgrams: List[str],
        cgrams: List[str],
        rgramslist: List[List[str]],
        numref: int,
    ) -> Tuple[float, float, float]:
        rgramsall = [rgram for rgrams in rgramslist for rgram in rgrams]
        rgramcounter = Counter(rgramsall)

        sgramcounter = Counter(sgrams)
        sgramcounter_rep = Counter()
        for sgram, scount in sgramcounter.items():
            sgramcounter_rep[sgram] = scount * numref

        cgramcounter = Counter(cgrams)
        cgramcounter_rep = Counter()
        for cgram, ccount in cgramcounter.items():
            cgramcounter_rep[cgram] = ccount * numref

        # KEEP
        keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
        keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
        keepgramcounterall_rep = sgramcounter_rep & rgramcounter

        keeptmpscore1 = 0
        keeptmpscore2 = 0
        for keepgram in keepgramcountergood_rep:
            keeptmpscore1 += (
                keepgramcountergood_rep[keepgram] / keepgramcounter_rep[keepgram]
            )
            # Fix an alleged bug [2] in the keep score computation.
            # keeptmpscore2 += keepgramcountergood_rep[keepgram] / keepgramcounterall_rep[keepgram]
            keeptmpscore2 += keepgramcountergood_rep[keepgram]
        # Define 0/0=1 instead of 0 to give higher scores for predictions that match
        #    a target exactly.
        keepscore_precision = 1
        keepscore_recall = 1
        if len(keepgramcounter_rep) > 0:
            keepscore_precision = keeptmpscore1 / len(keepgramcounter_rep)
        if len(keepgramcounterall_rep) > 0:
            # Fix an alleged bug [2] in the keep score computation.
            # keepscore_recall = keeptmpscore2 / len(keepgramcounterall_rep)
            keepscore_recall = keeptmpscore2 / sum(keepgramcounterall_rep.values())
        keepscore = 0
        if keepscore_precision > 0 or keepscore_recall > 0:
            keepscore = (
                2
                * keepscore_precision
                * keepscore_recall
                / (keepscore_precision + keepscore_recall)
            )

        # DELETION
        delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
        delgramcountergood_rep = delgramcounter_rep - rgramcounter
        delgramcounterall_rep = sgramcounter_rep - rgramcounter
        deltmpscore1 = 0
        deltmpscore2 = 0
        for delgram in delgramcountergood_rep:
            deltmpscore1 += (
                delgramcountergood_rep[delgram] / delgramcounter_rep[delgram]
            )
            deltmpscore2 += (
                delgramcountergood_rep[delgram] / delgramcounterall_rep[delgram]
            )
        # Define 0/0=1 instead of 0 to give higher scores for predictions that match
        #    a target exactly.
        delscore_precision = 1
        delscore_recall = 1
        if len(delgramcounter_rep) > 0:
            delscore_precision = deltmpscore1 / len(delgramcounter_rep)
        if len(delgramcounterall_rep) > 0:
            delscore_recall = deltmpscore1 / len(delgramcounterall_rep)
        delscore = 0
        if delscore_precision > 0 or delscore_recall > 0:
            delscore = (
                2
                * delscore_precision
                * delscore_recall
                / (delscore_precision + delscore_recall)
            )

        # ADDITION
        addgramcounter = set(cgramcounter) - set(sgramcounter)
        addgramcountergood = set(addgramcounter) & set(rgramcounter)
        addgramcounterall = set(rgramcounter) - set(sgramcounter)

        addtmpscore = 0
        for addgram in addgramcountergood:
            addtmpscore += 1

        # Define 0/0=1 instead of 0 to give higher scores for predictions that match
        #    a target exactly.
        addscore_precision = 1
        addscore_recall = 1
        if len(addgramcounter) > 0:
            addscore_precision = addtmpscore / len(addgramcounter)
        if len(addgramcounterall) > 0:
            addscore_recall = addtmpscore / len(addgramcounterall)
        addscore = 0
        if addscore_precision > 0 or addscore_recall > 0:
            addscore = (
                2
                * addscore_precision
                * addscore_recall
                / (addscore_precision + addscore_recall)
            )

        return (keepscore, delscore_precision, addscore)

    def SARIsent(self, ssent: str, csent: str, rsents: List[str]) -> float:
        numref = len(rsents)

        s1grams = ssent.split(" ")
        c1grams = csent.split(" ")
        s2grams = []
        c2grams = []
        s3grams = []
        c3grams = []
        s4grams = []
        c4grams = []

        r1gramslist = []
        r2gramslist = []
        r3gramslist = []
        r4gramslist = []
        for rsent in rsents:
            r1grams = rsent.split(" ")
            r2grams = []
            r3grams = []
            r4grams = []
            r1gramslist.append(r1grams)
            for i in range(0, len(r1grams) - 1):
                if i < len(r1grams) - 1:
                    r2gram = r1grams[i] + " " + r1grams[i + 1]
                    r2grams.append(r2gram)
                if i < len(r1grams) - 2:
                    r3gram = r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2]
                    r3grams.append(r3gram)
                if i < len(r1grams) - 3:
                    r4gram = (
                        r1grams[i]
                        + " "
                        + r1grams[i + 1]
                        + " "
                        + r1grams[i + 2]
                        + " "
                        + r1grams[i + 3]
                    )
                    r4grams.append(r4gram)
            r2gramslist.append(r2grams)
            r3gramslist.append(r3grams)
            r4gramslist.append(r4grams)

        for i in range(0, len(s1grams) - 1):
            if i < len(s1grams) - 1:
                s2gram = s1grams[i] + " " + s1grams[i + 1]
                s2grams.append(s2gram)
            if i < len(s1grams) - 2:
                s3gram = s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2]
                s3grams.append(s3gram)
            if i < len(s1grams) - 3:
                s4gram = (
                    s1grams[i]
                    + " "
                    + s1grams[i + 1]
                    + " "
                    + s1grams[i + 2]
                    + " "
                    + s1grams[i + 3]
                )
                s4grams.append(s4gram)

        for i in range(0, len(c1grams) - 1):
            if i < len(c1grams) - 1:
                c2gram = c1grams[i] + " " + c1grams[i + 1]
                c2grams.append(c2gram)
            if i < len(c1grams) - 2:
                c3gram = c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2]
                c3grams.append(c3gram)
            if i < len(c1grams) - 3:
                c4gram = (
                    c1grams[i]
                    + " "
                    + c1grams[i + 1]
                    + " "
                    + c1grams[i + 2]
                    + " "
                    + c1grams[i + 3]
                )
                c4grams.append(c4gram)

        (keep1score, del1score, add1score) = self.SARIngram(
            s1grams, c1grams, r1gramslist, numref
        )
        (keep2score, del2score, add2score) = self.SARIngram(
            s2grams, c2grams, r2gramslist, numref
        )
        (keep3score, del3score, add3score) = self.SARIngram(
            s3grams, c3grams, r3gramslist, numref
        )
        (keep4score, del4score, add4score) = self.SARIngram(
            s4grams, c4grams, r4gramslist, numref
        )
        avgkeepscore = sum([keep1score, keep2score, keep3score, keep4score]) / 4
        avgdelscore = sum([del1score, del2score, del3score, del4score]) / 4
        avgaddscore = sum([add1score, add2score, add3score, add4score]) / 4
        finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3
        return finalscore

    def normalize(
        self,
        sentence: str,
        lowercase: bool = True,
        tokenizer: str = "13a",
        return_str: bool = True,
    ) -> List[str]:

        # Normalization is requried for the ASSET dataset to allow using space
        # to split the sentence. Even though Wiki-Auto and TURK datasets,
        # do not require normalization, we do it for consistency.
        # Code adapted from the EASSE library [1] written by the authors of the ASSET dataset.
        # [1] https://github.com/feralvam/easse/blob/580bba7e1378fc8289c663f864e0487188fe8067/easse/utils/preprocessing.py#L7

        if lowercase:
            sentence = sentence.lower()

        sentence = self.tokenize(sentence, tokenizer)

        if not return_str:
            sentence = sentence.split()

        return sentence

    def tokenize(self, sentence: str, tokenizer: str) -> List[str]:
        if tokenizer in ["intl", "13a"]:
            sentence = sacrebleu.metrics.bleu._get_tokenizer(tokenizer)()(sentence)
        elif tokenizer == "moses":
            sentence = sacremoses.MosesTokenizer().tokenize(
                sentence, return_str=True, escape=False
            )
        elif tokenizer == "penn":
            sentence = sacremoses.MosesTokenizer().penn_tokenize(
                sentence, return_str=True
            )

        return sentence
