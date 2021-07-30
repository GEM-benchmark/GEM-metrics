#!/usr/bin/env python3

import functools
from gem_metrics.config import get_language_for_dataset, get_task_type_for_dataset
from typing import List, Optional
import json
import string
from pycountry import languages
from logzero import logger
from .tokenize import default_tokenize_func


class Texts:
    """Holder class for output texts or references."""

    PUNCTUATION = set(string.punctuation)

    def __init__(self, data_key, data, language="en"):
        self.data_key = data_key
        # TODO allow other data formats.
        if not isinstance(data, dict):
            self.filename = data
            with open(data, "r", encoding="UTF-8") as fh:
                data = json.load(fh)
                self.all_data = data
            if isinstance(data, dict) and "values" in data:
                self.all_data = data["values"]
        else:
            self.filename = data.get("filename")
            self.all_data = data["values"]
        logger.info(
            f"Loading {self.__class__.__name__.lower()} for {str(self.filename)}"
        )

        self.language = languages.get(alpha_2=language)

        # Allow bare lists of strings as well as lists of dicts.
        # In case of Dicts, check for IDs we need to shuffle.
        self.ids = None
        self.parent_ids = None
        if isinstance(self.all_data[0], str):
            self.data = [item for item in self.all_data]
        else:
            self.data = [item[data_key] for item in self.all_data]
            if "gem_id" in self.all_data[0].keys():
                self.ids = [item["gem_id"] for item in self.all_data]

            if "gem_parent_id" in self.all_data[0].keys():
                self.parent_ids = [item["gem_parent_id"] for item in self.all_data]

        # detect if we're using multiple texts per instance
        self.multi_ref = isinstance(self.data[0], list)
        # tokenize & keep a list and a whitespace version
        self.tokenize_func = default_tokenize_func(self.language)

    @functools.cached_property
    def untokenized(self):
        """Return list of (lists of) untokenized strings."""
        return self.data

    @functools.cached_property
    def _tokenized(self):
        """Return list of (lists of) untokenized strings."""
        if self.multi_ref:
            return [[self.tokenize_func(i) for i in inst] for inst in self.data]
        else:
            return [self.tokenize_func(ref) for ref in self.data]

    @functools.cached_property
    def whitespace_tokenized(self):
        """Return list of (lists of) tokenized strings (tokens separated by space)."""
        if self.multi_ref:
            return [[" ".join(i) for i in inst] for inst in self._tokenized]
        else:
            return [" ".join(ref) for ref in self._tokenized]

    @functools.cached_property
    def list_tokenized(self):
        """Return list of (lists of) lists of tokens."""
        return self._tokenized

    @functools.cached_property
    def list_tokenized_lower(self):
        """Return list of (lists of) lists of tokens, lowercased."""
        if self.multi_ref:
            return [
                [[w.lower() for w in ref] for ref in inst] for inst in self._tokenized
            ]
        else:
            return [[w.lower() for w in ref] for ref in self._tokenized]

    @functools.cached_property
    def list_tokenized_lower_nopunct(self):
        """Return list of (lists of) lists of tokens, lowercased, excluding punctuation."""
        if self.multi_ref:
            return [
                [[w for w in ref if w not in self.PUNCTUATION] for ref in inst]
                for inst in self.list_tokenized_lower
            ]
        else:
            return [
                [w for w in ref if w not in self.PUNCTUATION]
                for ref in self.list_tokenized_lower
            ]

    def assign_ids_and_unscramble(self, id_list: List):
        """Overwrite self.ids with id_list, unscramble and filter data.

        If the gem_id field is set, ensure that they appear in the same order as the
        original dataset. Can also be used to filter a dataset by ID.

        Args:
            id_list: ordered ID list of the associated reference file.
        """
        if self.ids is not None and id_list is not None:
            # Is the ID list set, but in a different order?
            if self.ids != id_list:
                logger.info(
                    "ID list is already set for %s and shuffled. Deshuffling and filtering now..."
                    % self.filename
                )
                # First, construct O(1) lookup.
                output_lookup = {
                    gem_id: example for gem_id, example in zip(self.ids, self.data)
                }
                # Then overwrite data with ordered version.
                self.data = [output_lookup[ordered_id] for ordered_id in id_list]
                self.ids = id_list
        else:
            # In this case we simply assume that the predictions were in order.
            # There is no other way to test for this.
            if id_list is not None:
                self.ids = id_list
            # when everything is None, we just make caching work
            elif self.ids is None:
                self.ids = ["generated-%05d" % i for i in range(len(self))]

    def __len__(self):
        return len(self.data)


class Predictions(Texts):
    """Data holder class for system outputs/predictions."""

    def __init__(self, data, language="en", task="agnostic"):
        # Task is used in QuestEval metric to select the correct model.
        self.task = task
        super().__init__(data_key="generated", data=data, language=language)
        #if self.ids is None:
        #    self.ids = ["unk-%05d" % i for i in range(len(self))]


class References(Texts):
    """Data holder class for references/targets."""

    def __init__(self, data, language="en"):
        super().__init__(data_key="target", data=data, language=language)

    @property
    def has_parent_ids(self):
        return self.parent_ids is not None


class Sources(Texts):
    """Data holder class for sources."""

    def __init__(self, data, language="en"):
        super().__init__(data_key="source", data=data, language=language)


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
        for dataset_name, data in self.all_data["tasks"].items():
            data["filename"] = self.name + "/" + dataset_name
            # Create Predictions with correct language - default to en.
            # Also change dashes to underscores since that is a common error.
            self.entries[dataset_name.replace("-", "_")] = Predictions(
                data, language=get_language_for_dataset(dataset_name),
                task=get_task_type_for_dataset(dataset_name)
            )

    def predictions_for(self, dataset_name: str) -> Optional[Predictions]:
        """Return per-dataset predictions"""
        return self.entries.get(dataset_name)

    @property
    def datasets(self):
        """List of datasets for which there are predictions available."""
        return list(self.entries.keys())
