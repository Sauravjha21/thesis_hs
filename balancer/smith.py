from collections.abc import Iterable

import numpy as np

from .base import Balancer


class Smith(Balancer):
    def __init__(self, q=0.5, rho=2, constant=1, **kwargs) -> None:
        if isinstance(q, float):
            q = q if q < 0.5 else 1 - q
            self.qs = [1 - q, q]
            self.classes = [0, 1]
        elif isinstance(q, Iterable):
            self.qs = [pr / sum(q) for pr in q]
            self.classes = [i for i, q in enumerate(self.qs)]
        else:
            raise ValueError("bad q")
        self.K = len(self.qs)
        self.assigned = np.array([0] * (self.K))
        self.n_observed = 0
        self.B = np.ones((self.K, self.K)) * constant - np.eye(self.K) * rho
        self.K = len(self.qs) - 1

    def assign_next(self, X: np.ndarray) -> np.ndarray:
        if self.n_observed == 0:
            pr = self.qs
        else:
            pr = self.qs + self.B @ (self.assigned / self.n_observed - self.qs)
            pr = np.clip(pr, 0, 1)
            pr = pr / np.sum(pr)
        assignment = np.random.choice(self.classes, p=pr)
        self.assigned[assignment] += 1
        self.n_observed += 1
        return assignment
