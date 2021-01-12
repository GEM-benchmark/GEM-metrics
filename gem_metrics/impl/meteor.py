#!/usr/bin/env python3

# Adapted from:
# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import subprocess
import threading
import urllib
import tarfile
from logzero import logger

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
METEOR_URL = 'https://github.com/GEM-benchmark/GEM-metrics/releases/download/data/meteor.tar.gz'


class PyMeteorWrapper:

    def __init__(self, language):
        """Try to instantiate and run METEOR. Will raise exceptions in case of errors."""
        self.my_dir = os.path.dirname(os.path.abspath(__file__))
        self.language = language
        self.check_meteor()
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR,
                           '-', '-', '-stdio', '-l', self.language, '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd,
                                         cwd=self.my_dir,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def check_meteor(self):
        """This will raise exceptions if we can't run Java or can't access the METEOR JAR file."""
        # check that we can actually run Java
        # we don't care what output we get, just that it doesn't fail
        subprocess.check_output(['java', '-version'], stderr=subprocess.STDOUT)

        # check and download meteor
        if not os.path.isfile(os.path.join(self.my_dir, METEOR_JAR)):
            logger.warn(f'METEOR not found at {self.my_path} -- downloading. This may take a few minutes.')
            tmp_fname, _ = urllib.request.urlretrieve(METEOR_URL)
            logger.warn(f'Extracting meteor to {self.my_path}')
            tmp_tgz = tarfile.open(tmp_fname, 'r:gz')
            tmp_tgz.extractall(path=self.my_dir)

    def compute_score(self, predictions, references):
        assert(len(predictions) == len(references))
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for pred, refs in zip(predictions, references):
            stat = self._stat(pred, refs)
            eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write('{}\n'.format(eval_line).encode('UTF-8'))
        self.meteor_p.stdin.flush()
        for _ in range(len(predictions)):
            scores.append(float(self.meteor_p.stdout.readline().decode('UTF-8').strip()))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return score, scores

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode('UTF-8'))
        self.meteor_p.stdin.flush()
        res = self.meteor_p.stdout.readline().decode('UTF-8').strip()
        return res

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode('UTF-8'))
        self.meteor_p.stdin.flush()
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats
        self.meteor_p.stdin.write('{}\n'.format(eval_line).encode('UTF-8'))
        self.meteor_p.stdin.flush()
        score = float(self.meteor_p.stdout.readline().decode('UTF-8').strip())
        # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
        # thanks for Andrej for pointing this out
        score = float(self.meteor_p.stdout.readline().decode('UTF-8').strip())
        self.lock.release()
        return score

    def __del__(self):
        if hasattr(self, 'lock') and self.lock:
            self.lock.acquire()
        if hasattr(self, 'meteor_p') and self.meteor_p:
            self.meteor_p.stdin.close()
            self.meteor_p.kill()
            self.meteor_p.wait()
        if hasattr(self, 'lock') and self.lock:
            self.lock.release()
