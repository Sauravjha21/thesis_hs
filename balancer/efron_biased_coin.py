import numpy as np

from .base import Balancer


class Efron(Balancer):
    def __init__(self, q=0.5, correction=2, **kwargs) -> None:
        q = q if q >= 0.5 else 1 - q
        odds = q / (1 - q)
        self.q = q
        self.q_plus = (odds * correction) / (odds * correction + 1)
        self.q_minus = (odds / correction) / (odds / correction + 1)

    def assign_all(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        a = np.array([0] * n)
        diff = 0
        for i in range(n):
            if diff == 0:
                pr = self.q
            elif diff > 0:
                pr = self.q_minus
            else:
                pr = self.q_plus
            a[i] = np.random.choice([0, 1], p=[1 - pr, pr])
            diff += (a[i] / self.q - (1 - a[i]) / (1 - self.q)) / 2
        return a
