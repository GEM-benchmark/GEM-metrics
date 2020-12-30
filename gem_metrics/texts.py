#!/usr/bin/env python3

import nltk
import os
import json

import pudb; pu.db
_NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nltk_data')
os.makedirs(_NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.insert(0, _NLTK_DATA_PATH)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=_NLTK_DATA_PATH)


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
                              for item in self.data]

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
        super().__init__(key='generated', tokenize_func=nltk.tokenize.word_tokenize)


class References(Texts):

    def __init__(self, data_file):
        super().__init__(key='target', tokenize_func=nltk.tokenize.word_tokenize)
