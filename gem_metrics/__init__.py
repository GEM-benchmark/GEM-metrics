#!/usr/bin/env python3


from .texts import Predictions, References

from .meteor import Meteor
from .bleu import BLEU

REFERENCED_METRICS = [BLEU, Meteor]
REFERENCELESS_METRICS = []
