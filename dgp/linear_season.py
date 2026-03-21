#! /usr/bin/python3

import numpy as np

from .base import DGP


class LinearSeasonDGP(DGP):
    def __init__(
        self,
        N=100,
        pate=1,
        ice_sd=0,
        noise_sd=0.1,
        num_covariates=4,
        period=1,
        num_treatments=1,
    ):
        super().__init__(N=N, D=num_covariates, K=num_treatments)
        tau = np.random.normal(size=N, loc=pate, scale=ice_sd)
        mu = np.sin(2 * np.pi * np.linspace(0, period, num=N)).reshape(-1, 1)
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

    @property
    def X(self) -> np.ndarray:
        return self._X


class LinearSeasonFactory(object):
    def __init__(self, N, period=1, K=1, pate=1):
        self.N = N
        self.period = period
        self.K = K
        self.pate = pate

    def create_dgp(self):
        return LinearSeasonDGP(
            N=self.N, pate=self.pate, period=self.period, num_treatments=self.K
        )
