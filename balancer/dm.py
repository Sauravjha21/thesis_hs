import numpy as np

from .base import Balancer


class DM(Balancer):
    """Implementation of Algorithm 2 from
    "Kernel Thinning", Dwivedi & Mackey (2021)
    """

    def __init__(self, N, D, delta=0.05, q=0.5, intercept=True):
        """
        Args:
            N (int): total number of points
            D (int): dimension of the data
            delta (float): probability of success
        """
        self.q = q
        # use the constant from corrollary 2 in Appendix L
        # self.alpha = np.log(4 * N * D / delta)
        self.intercept = intercept
        self.delta = delta
        self.D = D + int(self.intercept)
        self.alpha = 0.5 + np.log(4 * N / delta)
        self.w_i = np.zeros((self.D,))
        self.value_plus = 2 * (1 - self.q)
        self.value_minus = -2 * self.q

    def assign_next(self, x):
        dot = x @ self.w_i
        if dot > self.alpha:
            assignment = -1
            self.w_i += self.value_minus * x * abs(dot / self.alpha)
        elif dot < -self.alpha:
            assignment = 1
            self.w_i += self.value_plus * x * abs(dot / self.alpha)
        else:
            p_i = self.q * (1 - dot / self.alpha)

            if np.random.rand() < p_i:
                value = self.value_plus
                assignment = 1
            else:
                value = self.value_minus
                assignment = -1
            self.w_i += value * x
        return int((assignment + 1) / 2)

    def assign_all(self, X):
        if self.intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        X = X / np.sqrt(np.power(X, 2).sum(axis=1).max())
        return np.array([self.assign_next(X[i, :]) for i in range(X.shape[0])])


class OnlineDM(Balancer):
    """DM Balancer using doubling to accommodate a purely online strategy
    without a priori knowledge of sample size
    """

    def __init__(self, N, D, delta):
        self.cur_log_2 = 1
        self.dm = DM(2**self.cur_log_2, D, delta)
        self.assigned = 0
        self.w_i = np.zeros((D, 1))
        self.D = D

    def assign_next(self, x):
        if self.assigned == (2**self.cur_log_2):
            self.cur_log_2 = self.cur_log_2 + 1
            self.dm = DM(2**self.cur_log_2, self.dm.D, self.dm.delta)
        self.assigned += 1
        assignment = self.dm.assign_next(x)
        self.w_i += assignment * x.reshape(self.D, 1)
        return assignment
