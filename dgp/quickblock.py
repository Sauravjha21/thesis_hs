import numpy as np

from .base import DGP


class QuickBlockDGP(DGP):
    def __init__(self, N=100, K=2, pate=1, ite_sd=0, num_treatments=1):
        super().__init__(N=N, D=K, K=num_treatments)
        self._X = np.random.uniform(size=(N, K)) * 10
        e = np.random.normal(size=N)
        self._y = np.zeros((N, num_treatments + 1))
        self._y[:, 0] = np.prod(self._X, axis=1) + e
        for i in range(num_treatments):
            self._y[:, i + 1] = self._y[:, 0] + np.random.normal(
                size=N, loc=pate, scale=ite_sd
            )

    def Y(self, A: np.ndarray) -> np.ndarray:
        return self._y[range(len(A)), A]

    @property
    def X(self) -> np.ndarray:
        return self._X


class QuickBlockFactory:
    def __init__(self, N, K=1):
        self.N = N
        self.K = K

    def create_dgp(self):
        return QuickBlockDGP(N=self.N, num_treatments=self.K)
