from abc import ABCMeta, abstractmethod

import numpy as np


class DGP(metaclass=ABCMeta):
    def __init__(self, N: int, D: int, K: int = 1) -> None:
        self.n = N
        self.D = D
        self.K = K
        self._y = np.zeros((N, K + 1))

    @abstractmethod
    def Y(self, A: np.ndarray) -> np.ndarray:
        pass

    @property
    def potential_outcomes(self) -> np.ndarray:
        return self._y

    @property
    @abstractmethod
    def X(self) -> np.ndarray:
        pass

    def ATE(self) -> np.ndarray:
        return np.average(self.Y([1] * self.n) - self.Y([0] * self.n))
