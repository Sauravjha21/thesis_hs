#! /usr/bin/python3

import numpy as np

from .base import DGP


class LinearDriftDGP(DGP):
    def __init__(
        self,
        N=100,
        pate=1,
        ice_sd=0,
        noise_sd=0.1,
        num_covariates=4,
        slope=2,
        num_treatments=1,
    ):
        super().__init__(N=N, D=num_covariates, K=num_treatments)
        tau = np.random.normal(size=N, loc=pate, scale=ice_sd)
        mu = np.linspace(0, slope, num=N).reshape(-1, 1)
        mu = np.concatenate([mu] * num_covariates, axis=1)
        self._X = mu + np.random.normal(size=(N, num_covariates))
        self._X = self._X / np.sqrt(np.power(self._X, 2).sum(axis=1)).reshape(-1, 1)
        beta = np.random.uniform(size=(num_covariates, 1))
        self._y = np.zeros((N, num_treatments + 1))
        self._y[:, 0] = (self._X @ beta).reshape(-1)
        for i in range(num_treatments):
            self._y[:, i + 1] = (
                self._y[:, 0] + tau + np.random.normal(size=N, loc=0, scale=noise_sd)
            )
        self._y[:, 0] += np.random.normal(size=N, loc=0, scale=noise_sd)

    def Y(self, A: np.ndarray) -> np.ndarray:
        return self._y[range(len(A)), A]
        # return np.where(np.array(A).flatten() == 1, self.y1, self.y0)

    @property
    def X(self) -> np.ndarray:
        return self._X


class LinearDriftFactory:
    def __init__(self, N, slope=1, K=1):
        self.N = N
        self.slope = slope
        self.K = K

    def create_dgp(self):
        return LinearDriftDGP(N=self.N, slope=self.slope, num_treatments=self.K)
