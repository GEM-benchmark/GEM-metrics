#!/usr/bin/env python3


from .texts import Predictions, References

from .meteor import Meteor
from .bleu import BLEU
from .rouge import ROUGE

REFERENCED_METRICS = [BLEU, Meteor, ROUGE]
REFERENCELESS_METRICS = []
