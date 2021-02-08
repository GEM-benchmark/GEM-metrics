#!/usr/bin/env python3

import numpy as np
from nltk import ngrams

from .metric import ReferencelessMetric


class NGramStats(ReferencelessMetric):
    """Ngram basic statistics and entropy, working with tokenized & lowercased data (+ variant excluding punctuation):

    - data length (total number of words)
    - mean instance length (number of words)
    - distinct-N (ratio of distinct N-grams / total number of N-grams)
    - vocab_size-N (total number of distinct N-grams)
    - unique-N (number of N-grams that only occur once in the whole data)
    - entropy-N (Shannon entropy over N-grams)
    - cond-entropy-N (language model style conditional entropy -- N-grams conditioned on N-1-grams)

    All these are computed for 1,2,3-grams (conditional entropy only for 2,3).

    Based on:
    https://github.com/evanmiltenburg/NLG-diversity/blob/main/diversity.py
    https://github.com/tuetschek/e2e-stats/blob/master/nlg_dataset_stats.py
    """

    def compute(self, predictions):

        results = {}
        for data_id, data in [('', predictions.list_tokenized_lower), ('-nopunct', predictions.list_tokenized_lower_nopunct)]:

            lengths = [len(inst) for inst in data]
            results[f'total_length{data_id}'] = sum(lengths)
            results[f'mean_pred_length{data_id}'] = np.mean(lengths)
            results[f'std_pred_length{data_id}'] = np.std(lengths)
            results[f'median_pred_length{data_id}'] = np.median(lengths)
            results[f'min_pred_length{data_id}'] = min(lengths)
            results[f'max_pred_length{data_id}'] = max(lengths)

            last_ngram_freqs = None  # for conditional entropy, we need lower-level n-grams
            for N in [1, 2, 3]:
                ngram_freqs, uniq_ngrams, ngram_len = self._ngram_stats(data, N)
                results[f'distinct-{N}{data_id}'] = len(ngram_freqs) / ngram_len if ngram_len > 0 else 0
                results[f'vocab_size-{N}{data_id}'] = len(ngram_freqs)
                results[f'unique-{N}{data_id}'] = uniq_ngrams
                results[f'entropy-{N}{data_id}'] = self._entropy(ngram_freqs)

                if last_ngram_freqs:
                    results[f'cond_entropy-{N}{data_id}'] = self._cond_entropy(ngram_freqs, last_ngram_freqs)
                last_ngram_freqs = ngram_freqs

        return results

    def _ngram_stats(self, data, N):
        """Return basic ngram statistics, as well as a dict of all ngrams and their freqsuencies."""
        ngram_freqs = {}   # ngrams with frequencies
        ngram_len = 0  # total number of ngrams
        for inst in data:
            for ngram in ngrams(inst, N):
                ngram_freqs[ngram] = ngram_freqs.get(ngram, 0) + 1
                ngram_len += 1
        # number of unique ngrams
        uniq_ngrams = len([val for val in ngram_freqs.values() if val == 1])
        return ngram_freqs, uniq_ngrams, ngram_len

    def _entropy(self, ngram_freqs):
        """Shannon entropy over ngram frequencies"""
        total_freq = sum(ngram_freqs.values())
        return - sum([freq / total_freq * np.log2(freq / total_freq) for freq in ngram_freqs.values()])

    def _cond_entropy(self, joint, ctx):
        """Conditional/next-word entropy (language model style), using ngrams (joint) and n-1-grams (ctx)."""
        total_joint = sum(joint.values())
        total_ctx = sum(ctx.values())
        # H(y|x) = - sum_{x,y} p(x,y) log_2 p(y|x)
        # p(y|x) = p(x,y) / p(x)
        return - sum([freq / total_joint
                      * np.log2((freq / total_joint) / (ctx[ngram[:-1]] / total_ctx))
                      for ngram, freq in joint.items()])
