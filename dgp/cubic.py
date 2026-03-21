#! /usr/bin/python3

import numpy as np

from .base import DGP


class CubicDGP(DGP):
    def __init__(
        self, N=100, pate=1, ice_sd=0, noise_sd=0.1, num_covariates=2, num_treatments=1
    ):
        super().__init__(N=N, D=num_covariates, K=num_treatments)
        tau = np.random.normal(size=N, loc=pate, scale=ice_sd)
        self._X = np.random.uniform(low=-1, high=1, size=(N, num_covariates))
        self._X = self._X / np.sqrt(np.power(self._X, 2).sum(axis=1)).reshape(-1, 1)
        x1 = self._X[:, 0]
        x2 = self._X[:, 1]
        self._y = np.zeros((N, num_treatments + 1))
        self._y[:, 0] = (
            x1
            - x2
            + np.power(x1, 2)
            + np.power(x2, 2)
            - 2 * x1 * x2
            + np.power(x1, 3)
            - np.power(x2, 3)
            - 3 * np.power(x1, 2) * x2
            + 3 * x1 * np.power(x2, 2)
        )
        for i in range(num_treatments):
            self._y[:, i + 1] = (
                self._y[:, 0] + tau + np.random.normal(size=N, loc=0, scale=noise_sd)
            )
        self._y[:, 0] += np.random.normal(size=N, loc=0, scale=noise_sd)

    def Y(self, A: np.ndarray) -> np.ndarray:
        return self._y[range(len(A)), A]

    @property
    def X(self) -> np.ndarray:
        return self._X


class CubicFactory(object):
    def __init__(self, N, K=1, pate=1):
        self.N = N
        self.K = K
        self.pate = pate

    def create_dgp(self):
        return CubicDGP(N=self.N, pate=self.pate, num_treatments=self.K)
