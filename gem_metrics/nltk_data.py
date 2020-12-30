#!/usr/bin/env python3

import nltk
import os
import re

_NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'nltk_data')
os.makedirs(_NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.insert(0, _NLTK_DATA_PATH)


def nltk_ensure_download(package):
    """Check if the given package is available, download if needed."""
    try:
        nltk.data.find(package)
    except LookupError:
        package_id = re.sub('^[^/]*/', '', package)
        nltk.download(package_id, download_dir=_NLTK_DATA_PATH)
