#!/usr/bin/env python3

import nltk
import json
from .nltk_data import nltk_ensure_download



class Texts:
    """Holder class for output texts or references."""

    def __init__(self, key, data_file, tokenize_func):
        self.filename = data_file
        self.key = key
        # TODO allow other data formats?
        with open(data_file, 'r', encoding='UTF-8') as fh:
            self.all_data = json.load(fh)
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
    def detokenized(self):
        """Return list of (lists of) detokenized strings."""
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

    def __init__(self, data_file):
        nltk_ensure_download('tokenizers/punkt')
        super().__init__(key='generated', data_file=data_file, tokenize_func=nltk.tokenize.word_tokenize)


class References(Texts):

    def __init__(self, data_file):
        nltk_ensure_download('tokenizers/punkt')
        super().__init__(key='target', data_file=data_file, tokenize_func=nltk.tokenize.word_tokenize)
