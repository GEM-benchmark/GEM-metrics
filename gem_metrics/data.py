#!/usr/bin/env python3

import nltk
import os
import re
from logzero import logger
import sys
import time
from functools import partial
import urllib
import tarfile


_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data'))
_NLTK_DATA_PATH = os.path.join(_BASE_DIR, 'nltk_data')

os.makedirs(_NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.insert(0, _NLTK_DATA_PATH)


def nltk_ensure_download(package):
    """Check if the given package is available, download if needed."""
    try:
        nltk.data.find(package)
    except LookupError:
        package_id = re.sub('^[^/]*/', '', package)
        nltk.download(package_id, download_dir=_NLTK_DATA_PATH)


def _urlretrieve_reporthook(count, block_size, total_size, start_time):
    """Helper function -- progress indicator."""
    # adapted from https://stackoverflow.com/questions/51212/how-to-write-a-download-progress-indicator-in-python
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stderr.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stderr.flush()


def ensure_download(subdir, target_file, url):
    """Check if the given data file is available, download if needed."""

    target_dir = os.path.join(_BASE_DIR, subdir)
    target_file = os.path.join(target_dir, target_file)
    if not os.path.isfile(target_file):
        os.makedirs(target_dir, exist_ok=True)
        logger.warn(f'{target_file} not found -- downloading {url}. This may take a few minutes.')
        # tar.gz download
        if url.endswith('.tgz') or url.endswith('.tar.gz'):
            tmp_fname, _ = urllib.request.urlretrieve(url, reporthook=partial(_urlretrieve_reporthook, start_time=time.time()))
            sys.stderr.write("\n")
            logger.warn(f'Extracting from {tmp_fname} to {target_dir}')
            tmp_tgz = tarfile.open(tmp_fname, 'r:gz')
            tmp_tgz.extractall(target_dir)
        # single file download
        else:
            urllib.request.urlretrieve(url, target_file, reporthook=partial(_urlretrieve_reporthook, start_time=time.time()))
            sys.stderr.write("\n")
    return target_file
