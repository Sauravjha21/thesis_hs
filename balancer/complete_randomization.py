from collections.abc import Iterable
from functools import reduce
from operator import add

import numpy as np

from .base import Balancer


class Complete(Balancer):
    def __init__(self, q=0.5, **kwargs) -> None:
        if isinstance(q, float):
            q = q
            self.qs = [1 - q, q]
            self.classes = [0, 1]
        elif isinstance(q, Iterable):
            self.qs = [pr / sum(q) for pr in q]
            self.classes = [i for i, q in enumerate(self.qs)]
        else:
            raise ValueError("bad q")
        self.K = len(self.qs) - 1

    def assign_all(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        ns = [int(np.floor(n * q)) for q in self.qs]
        remainder = n - sum(ns)
        extra = np.random.choice(self.classes, size=remainder, replace=True, p=self.qs)
        ns = [n + sum(extra == i) for i, n in enumerate(ns)]
        treatment_vector = reduce(
            add, [[cl] * n for cl, n in zip(self.classes, ns, strict=False)]
        )
        return np.random.choice(treatment_vector, size=n, replace=False)
