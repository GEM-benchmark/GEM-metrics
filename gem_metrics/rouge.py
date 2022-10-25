#!/usr/bin/env python3

from .metric import ReferencedMetric
import numpy as np
from rouge_score import rouge_scorer, scoring


class ROUGE(ReferencedMetric):
    """ROUGE uses Google implementation (https://github.com/google-research/google-research/tree/master/rouge)
    but adds own implementation of multi-ref jackknifing.
    The Google implementation should be identical to Rouge-155 (except tokenization?),
    the jackknifing follows the description of the ROUGE paper.
    """

    def compute(self, cache, predictions, references):
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
        scores = {}
        # TODO expecting pretokenized data, do we want to imitate Rouge-155 tokenizer somehow?
        for refs, pred, pred_id in zip(
            references.whitespace_tokenized,
            predictions.whitespace_tokenized,
            predictions.ids,
        ):
            # ROUGE multi-ref jackknifing
            if len(refs) > 1:
                cur_scores = [rouge.score(ref, pred) for ref in refs]

                # get best score for all leave-one-out sets
                best_scores = []
                for leave in range(len(refs)):
                    cur_scores_leave_one = [
                        cur_scores[s] for s in range(len(refs)) if s != leave
                    ]
                    best_scores.append(
                        {
                            rouge_type: max(
                                [s[rouge_type] for s in cur_scores_leave_one],
                                key=lambda s: s.fmeasure,
                            )
                            for rouge_type in rouge_types
                        }
                    )

                # average the leave-one-out bests to produce the final score
                score = {
                    rouge_type: scoring.Score(
                        np.mean([b[rouge_type].precision for b in best_scores]),
                        np.mean([b[rouge_type].recall for b in best_scores]),
                        np.mean([b[rouge_type].fmeasure for b in best_scores]),
                    )
                    for rouge_type in rouge_types
                }
            else:
                score = rouge.score(refs[0], pred)

            # convert the named tuples to plain nested dicts
            score = {
                rouge_type: {
                    "precision": score[rouge_type].precision,
                    "recall": score[rouge_type].recall,
                    "fmeasure": score[rouge_type].fmeasure,
                }
                for rouge_type in rouge_types
            }
            # Write to cache if not None.
            if cache is not None:
                cache_key = (self.__class__.__name__, predictions.filename, pred_id)
                cache[cache_key] = score
            scores[pred_id] = score

        return scores
