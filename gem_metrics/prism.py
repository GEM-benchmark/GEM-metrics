#!/usr/bin/env python3
from repro.models.thompson2020 import Prism as _Prism

from .metric import ReproReferencedMetric


class Prism(ReproReferencedMetric):
    def __init__(self, language: str = "en", device: int = 0):
        metric = _Prism(language=language, device=device)
        super().__init__(metric)
