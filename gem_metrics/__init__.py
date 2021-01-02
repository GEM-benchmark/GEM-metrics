#!/usr/bin/env python3


# Data holder classes
from .texts import Predictions, References

# Metric implementations
from .meteor import Meteor
from .bleu import BLEU
from .rouge import ROUGE
from .msttr import MSTTR
from .ngrams import NGramStats

# Lists of metrics to use
# TODO make this populate automatically based on imports
REFERENCED_METRICS = [BLEU, Meteor, ROUGE]
REFERENCELESS_METRICS = [MSTTR, NGramStats]
