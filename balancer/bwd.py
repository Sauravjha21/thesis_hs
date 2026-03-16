import numpy as np

from .base import Balancer


class BWD(Balancer):
    """Implementation of our algorithm"""

    def __init__(self, N, D, delta=0.05, q=0.5, intercept=True, phi=1):
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
        self.N = N
        self.D = D + int(self.intercept)
        self.w_i = np.zeros((self.D,))
        self.value_plus = 2 * (1 - self.q)
        self.value_minus = -2 * self.q
        self.phi = phi
        self.set_alpha(self.N)
        self.iterations = 0

    def set_alpha(self, N):
        self.alpha = np.log(2 * N / self.delta) * min(1 / self.q, 9.32)

    def assign_next(self, x):
        # x_norm = np.sqrt(np.power(x, 2).sum().item())
        # if x_norm < 1.05:
        #     x_norm = 1.0
        dot = x @ self.w_i
        if abs(dot) > self.alpha:
            # print("RESTART")
            self.w_i = np.zeros((self.D,))
            self.set_alpha(self.N - self.iterations)
            dot = x @ self.w_i

        p_i = self.q * (1 - self.phi * dot / self.alpha)

        if np.random.rand() < p_i:
            value = self.value_plus
            assignment = 1
        else:
            value = self.value_minus
            assignment = -1
        self.w_i += value * x
        self.iterations += 1
        return int((assignment + 1) / 2)

    def assign_all(self, X):
        if self.intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        return np.array([self.assign_next(X[i, :]) for i in range(X.shape[0])])


class BWDRandom(Balancer):
    """Implementation of our algorithm"""

    def __init__(self, N, D, q=0.5, intercept=True, phi=1):
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
        self.delta = 1 / N
        self.N = N
        self.D = D + int(self.intercept)
        self.w_i = np.zeros((self.D,))
        self.value_plus = 2 * (1 - self.q)
        self.value_minus = -2 * self.q
        self.phi = max(phi, 1e-15)
        self.set_alpha(self.N)
        self.iterations = 0

    def set_alpha(self, N):
        self.alpha = np.log(2 * N / self.delta) * min(1 / self.q, 9.32)

    def assign_next(self, x):
        dot = x @ self.w_i
        if abs(dot) > self.alpha:
            self.w_i = np.zeros((self.D,))
            self.set_alpha(self.N - self.iterations)
            dot = x @ self.w_i

        p_i = self.q * (1 - self.phi * dot / self.alpha)

        if np.random.rand() < p_i:
            value = self.value_plus
            assignment = 1
        else:
            value = self.value_minus
            assignment = -1
        self.w_i += value * x
        self.iterations += 1
        return int((assignment + 1) / 2)

    def assign_all(self, X):
        if self.intercept:
            X = np.hstack((X, np.ones((X.shape[0], 1))))
        return np.array([self.assign_next(X[i, :]) for i in range(X.shape[0])])


class OnlineBWD(Balancer):
    """DM Balancer using doubling to accommodate a purely online strategy
    without a priori knowledge of sample size
    """

    def __init__(self, N, D, delta):
        self.cur_log_2 = 1
        self.dm = BWD(2**self.cur_log_2, D, delta)
        self.assigned = 0
        self.w_i = np.zeros((D, 1))
        self.D = D

    def assign_next(self, x):
        if self.assigned == (2**self.cur_log_2):
            self.cur_log_2 = self.cur_log_2 + 1
            self.dm = BWD(2**self.cur_log_2, self.dm.D, self.dm.delta)
        self.assigned += 1
        assignment = self.dm.assign_next(x)
        self.w_i += assignment * x.reshape(self.D, 1)
        return assignment
