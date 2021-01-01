#!/usr/bin/env python3

"""
!!! This is a (modified/shortened) copy of
https://github.com/evanmiltenburg/NLG-diversity/blob/main/diversity.py
"""

from string import punctuation
from nltk import ngrams
import random

random.seed(1234)
PUNCTUATION = set(punctuation)


def get_vocabulary(tokenized_data):
    "Compute vocabulary, based on the tokenized data."
    vocab = set()
    for sent in tokenized_data:
        vocab.update(sent)
    return vocab


def get_ngram_vocabulary(tokenized_data, n):
    "Compute n-gram vocabulary, based on the tokenized data."
    vocab = set()
    for sent in tokenized_data:
        vocab.update(ngrams(sent, n))
    return vocab


def TTR(list_of_words):
    "Compute type-token ratio."
    return len(set(list_of_words)) / len(list_of_words)


def MSTTR(tokenized_data, window_size):
    "Compute Mean-Segmental Type-Token Ratio (MSTTR; Johnson, 1944)."
    chunk = []
    ttrs = []
    for sentence in tokenized_data:
        chunk_length = len(chunk)
        sentence_length = len(sentence)
        combined = chunk_length + sentence_length
        if combined < window_size:
            chunk.extend(sentence)
        elif combined == window_size:
            chunk.extend(sentence)
            ttrs.append(TTR(chunk))
            chunk = []
        else:
            needed = window_size - chunk_length
            chunk.extend(sentence[:needed])
            ttrs.append(TTR(chunk))
            chunk = sentence[needed:]
    results = {'msttr_value': sum(ttrs) / len(ttrs),
               'num_ttrs': len(ttrs),
               'ttrs': ttrs}
    return results


def repeated_MSTTR(tokenized_data, window_size, repeats=5):
    "Repeated MSTTR to obtain a more robust MSTTR value."
    msttrs = []
    for i in range(repeats):
        sentences = random.sample(tokenized_data, len(tokenized_data))
        msttr_results = MSTTR(sentences, window_size)
        msttrs.append(msttr_results['msttr_value'])
    results = sum(msttrs) / len(msttrs)
    return results


def num_tokens(tokenized_data):
    "Compute the number of tokens."
    return sum(len(sentence) for sentence in tokenized_data)


def num_ngram_tokens(tokenized_data, n):
    "Compute the number of tokens."
    return sum(len(list(ngrams(sentence, n))) for sentence in tokenized_data)
