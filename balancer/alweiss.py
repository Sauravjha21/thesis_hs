import numpy as np

from .base import Balancer


class Alweiss(Balancer):
    """Implementation of Algorithm 1 from
    "DISCREPANCY MINIMIZATION VIA A SELF-BALANCING WALK", Alweiss, et al. (2020)
    """

    def __init__(self, N, D, delta=0.05):
        """
        Args:
            N (int): total number of points
            D (int): dimension of the data
            delta (float): probability of success
        """
        # normalizing constant defined by Alweiss
        self.c = 30 * np.log(N * D / delta)
        self.delta = delta
        self.D = D
        self.w_i = np.zeros((D,))

    def assign_next(self, x):
        # Line four
        if (np.abs(x @ self.w_i) > self.c) or (np.max(self.w_i) > self.c):
            p_i = 0.5  # fallback to CR rather than failing
        else:
            p_i = 0.5 - ((x @ self.w_i) / (2 * self.c))
        assignment = 1 if np.random.rand() < p_i else -1
        self.w_i += assignment * x
        return int((assignment + 1) / 2)

    def assign_all(self, X):
        X = X / np.sqrt(np.power(X, 2).sum(axis=1).max())
        return np.array([self.assign_next(X[i, :]) for i in range(X.shape[0])])


class OnlineAlweiss(Balancer):
    """Alweiss Balancer using doubling to accommodate a purely online strategy
    without a priori kno
    """

    def __init__(self, N, D, delta):
        self.cur_log_2 = 1
        self.alweiss = Alweiss(2**self.cur_log_2, D, delta)
        self.assigned = 0
        self.w_i = np.zeros((D, 1))
        self.D = D

    def assign_next(self, x):
        if self.assigned == (2**self.cur_log_2):
            self.cur_log_2 = self.cur_log_2 + 1
            self.alweiss = Alweiss(
                2**self.cur_log_2, self.alweiss.D, self.alweiss.delta
            )
        self.assigned += 1
        assignment = self.alweiss.assign_next(x)
        self.w_i += assignment * x.reshape(self.D, 1)
        return assignment

    def assign_all(self, X):
        X = X / np.sqrt(np.power(X, 2).sum(axis=1).max())
        return np.array([self.assign_next(X[i, :]) for i in range(X.shape[0])])
