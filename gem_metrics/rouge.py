#!/usr/bin/env python3

from .metric import ReferencedMetric
import numpy as np
from rouge_score import rouge_scorer, scoring


class ROUGE(ReferencedMetric):
    """ROUGE uses Google implementation (https://github.com/google-research/google-research/tree/master/rouge)
    but adds own implementation of multi-ref jackknifing, which doesn't seem to be supported by the
    Google code.
    The Google implementation should be identical to Rouge-155 (except tokenization?), the jackknifing is
    implemented after the ROUGE paper.
    """

    def compute(self, predictions, references):
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()
        # TODO expecting pretokenized data, do we want to imitate Rouge-155 tokenizer somehow?
        for refs, pred in zip(references.whitespace_tokenized, predictions.whitespace_tokenized):

            # ROUGE multi-ref jackknifing
            if len(refs) > 1:
                scores = []
                for ref in refs:
                    scores.append(rouge.score(ref, pred))

                # get best score for all leave-one-out sets
                best_scores = []
                for leave in range(len(refs)):
                    cur_scores = [s for s in scores]
                    del cur_scores[leave]
                    best_scores.append({rouge_type: max([s[rouge_type] for s in cur_scores],
                                                        key=lambda s: s.fmeasure)
                                        for rouge_type in rouge_types})

                # average the leave-one-out bests to produce the final score
                score = {rouge_type: scoring.Score(np.mean([b[rouge_type].precision for b in best_scores]),
                                                   np.mean([b[rouge_type].recall for b in best_scores]),
                                                   np.mean([b[rouge_type].fmeasure for b in best_scores]))
                         for rouge_type in rouge_types}
            else:
                score = rouge.score(refs[0], pred)
            aggregator.add_scores(score)

        result = aggregator.aggregate()
        # convert the named tuples to plain nested dicts
        result = {rouge_type: {vtype: dict(val._asdict()) for vtype, val in result[rouge_type]._asdict().items()}
                  for rouge_type in rouge_types}
        return result
