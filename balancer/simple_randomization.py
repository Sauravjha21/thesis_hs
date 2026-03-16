from collections.abc import Iterable

import numpy as np

from .base import Balancer


class Simple(Balancer):
    def __init__(self, q=0.5, **kwargs) -> None:
        if isinstance(q, float):
            q = q if q < 0.5 else 1 - q
            self.qs = [1 - q, q]
            self.classes = [0, 1]
        elif isinstance(q, Iterable):
            self.qs = q
            self.classes = [i for i, q in enumerate(self.qs)]
        self.K = len(self.qs) - 1

    def assign_all(self, X: np.ndarray) -> np.ndarray:
        return np.random.choice(self.classes, size=X.shape[0], replace=True, p=self.qs)
