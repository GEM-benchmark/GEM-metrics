#!/usr/bin/env python3

from typing import Optional
import json
import string
from pycountry import languages
from logzero import logger
from .tokenize import default_tokenize_func


class Texts:
    """Holder class for output texts or references."""

    PUNCTUATION = set(string.punctuation)

    def __init__(self, data_key, data):
        self.data_key = data_key
        # TODO allow other data formats.
        if not isinstance(data, dict):
            self.filename = data
            with open(data, "r", encoding="UTF-8") as fh:
                data = json.load(fh)
                self.all_data = data
        else:
            self.filename = data.get("filename")
            self.all_data = data["values"]
        logger.info(
            f"Loading {self.__class__.__name__.lower()} for {str(self.filename)}"
        )

        # TODO: take out this hack again.
        self.language = languages.get(alpha_2='en')

        # if len(data["language"]) > 3 or data["language"][0].isupper():
        #     self.language = languages.get(name=data["language"])
        # elif len(data["language"]) == 3:
        #     self.language = languages.get(alpha_3=data["language"])
        # else:
        #     self.language = languages.get(alpha_2=data["language"])

        # Allow bare lists of strings as well as lists of dicts.
        # In case of Dicts, check for IDs we need to shuffle.
        self.ids = None
        if isinstance(self.all_data[0], str):
            self.data = [item for item in self.all_data]
        else:
            self.data = [item[data_key] for item in self.all_data]
            if "gem_id" in self.all_data[0].keys():
                self.ids = [item["gem_id"] for item in self.all_data]


        # detect if we're using multiple texts per instance
        self.multi_ref = isinstance(self.data[0], list)
        # tokenize & keep a list and a whitespace version
        tokenize_func = default_tokenize_func(self.language)
        if self.multi_ref:
            self._tokenized = [[tokenize_func(i) for i in inst] for inst in self.data]
            self._ws_tokenized = [
                [" ".join(i) for i in inst] for inst in self._tokenized
            ]
            self._lc_tokenized = [
                [[w.lower() for w in ref] for ref in inst] for inst in self._tokenized
            ]
            self._nopunct_lc_tokenized = [
                [[w for w in ref if w not in self.PUNCTUATION] for ref in inst]
                for inst in self._lc_tokenized
            ]
        else:
            self._tokenized = [tokenize_func(ref) for ref in self.data]
            self._ws_tokenized = [" ".join(ref) for ref in self._tokenized]
            self._lc_tokenized = [[w.lower() for w in ref] for ref in self._tokenized]
            self._nopunct_lc_tokenized = [
                [w for w in ref if w not in self.PUNCTUATION]
                for ref in self._lc_tokenized
            ]

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

    @property
    def list_tokenized_lower(self):
        """Return list of (lists of) lists of tokens, lowercased."""
        return self._lc_tokenized

    @property
    def list_tokenized_lower_nopunct(self):
        """Return list of (lists of) lists of tokens, lowercased, excluding punctuation."""
        return self._nopunct_lc_tokenized

    def __len__(self):
        return len(self.data)

    def assign_ids(self, id_list: list):
        """Overwrite self.ids with the list after checking that the length is okay."""
        assert len(self.data) == len(
            id_list
        ), "References have different length as generated texts."
        assert self.ids is None, "ID list is already set for %s" % self.filename
        self.ids = id_list


class Predictions(Texts):
    """Data holder class for system outputs/predictions."""

    def __init__(self, data):
        # Task is used in QuestEval metric to select correct model.
        self.task = data.get("task", "agnostic")
        super().__init__(data_key="generated", data=data)


class References(Texts):
    """Data holder class for references/targets."""

    def __init__(self, data):
        super().__init__(data_key="target", data=data)


class Sources(Texts):
    """Data holder class for sources."""

    def __init__(self, data):
        super().__init__(data_key="source", data=data)


class Submission:
    """Data class for multiple submissions."""

    def __init__(self, data):
        if isinstance(data, dict):
            self.all_data = data
        else:
            self.filename = data
            with open(data, "r", encoding="UTF-8") as fh:
                self.all_data = json.load(fh)
        self.name = data["submission_name"]
        self.param_count = data.get("param_count")
        if not self.param_count:
            logger.warn("Model parameter count not present in the submission file.")
        self.entries = {}
        for key, data in self.all_data["tasks"].items():
            data["filename"] = self.name + "/" + key
            self.entries[key] = Predictions(data)

    def predictions_for(self, dataset_name: str) -> Optional[Predictions]:
        """Return per-dataset predictions"""
        return self.entries.get(dataset_name)

    @property
    def datasets(self):
        """List of datasets for which there are predictions available."""
        return list(self.entries.keys())
