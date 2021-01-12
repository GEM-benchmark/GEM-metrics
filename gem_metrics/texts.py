#!/usr/bin/env python3

from typing import Optional
import json
import string
import nltk
from .nltk_data import nltk_ensure_download


class Texts:
    """Holder class for output texts or references."""

    def __init__(self, key, data, tokenize_func):
        self.key = key
        if not isinstance(data, dict):
            self.filename = data
            # TODO allow other data formats?
            with open(data, 'r', encoding='UTF-8') as fh:
                data = json.load(fh)
        else:
            self.filename = data.get('filename')
        self.all_data = data['values']
        self.language = data['language']

        self.data = [item[key] for item in self.all_data]

        # detect if we're using multiple texts per instance
        self.multi_ref = isinstance(self.data[0], list)
        # tokenize & keep a list and a whitespace version
        self._tokenized = [([tokenize_func(i) for i in item]
                            if self.multi_ref
                            else tokenize_func(item))
                           for item in self.data]
        self._ws_tokenized = [([' '.join(i) for i in item]
                               if self.multi_ref
                               else ' '.join(item))
                              for item in self._tokenized]

    @property
    def untokenized(self):
        """Return list of (lists of) untokenized strings."""
        return self.data

    @property
    def whitespace_tokenized(self):
        """Return list of (lists of) tokenized strings (tokens separated by space)."""
        return self._ws_tokenized

    @property
    def list_tokenized(self):
        """Return list of (lists of) lists of tokens."""
        return self._tokenized

    def __len__(self):
        return len(self.data)


class Predictions(Texts):
    """Data holder class for system outputs/predictions."""

    PUNCTUATION = set(string.punctuation)

    def __init__(self, data):
        nltk_ensure_download('tokenizers/punkt')
        super().__init__(key='generated', data=data, tokenize_func=nltk.tokenize.word_tokenize)
        self._lc_tokenized = [[w.lower() for w in item] for item in self.list_tokenized]
        self._nopunct_lc_tokenized = [[w for w in item if w not in self.PUNCTUATION] for item in self._lc_tokenized]

    @property
    def list_tokenized_lower(self):
        """Return list of lists of tokens, lowercased."""
        return self._lc_tokenized

    @property
    def list_tokenized_lower_nopunct(self):
        """Return list of lists of tokens, lowercased, excluding punctuation."""
        return self._nopunct_lc_tokenized


class References(Texts):
    """Data holder class for references/targets."""

    def __init__(self, data):
        nltk_ensure_download('tokenizers/punkt')
        super().__init__(key='target', data=data, tokenize_func=nltk.tokenize.word_tokenize)


class Submission:
    """Data class for multiple submissions."""

    def __init__(self, data):
        if isinstance(data, dict):
            self.all_data = data
        else:
            self.filename = data
            with open(data, 'r', encoding='UTF-8') as fh:
                self.all_data = json.load(fh)
        self.name = data['submission_name']
        self.param_count = data.get('param_count')
        self.entries = {}
        for key in self.all_data.keys():
            if not key.endswith('_val'):
                continue
            dataset_name = key[:-4]
            self.entries[dataset_name] = Predictions(self.all_data[key])

    def predictions_for(self, dataset_name: str) -> Optional[Predictions]:
        """Return per-dataset predictions"""
        return self.entries.get(dataset_name)
