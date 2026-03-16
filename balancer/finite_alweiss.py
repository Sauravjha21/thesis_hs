import numpy as np

from .base import Balancer


class FiniteAlweiss(Balancer):
    """Implementation of Algorithm 1 from
    "DISCREPANCY MINIMIZATION VIA A SELF-BALANCING WALK", Alweiss, et al. (2020)
    """

    def __init__(self, N, D, delta=0.05, q=0.5):
        """
        Args:
            N (int): total number of points
            D (int): dimension of the data
            delta (float): probability of success
        """
        # normalizing constant defined by Alweiss
        self.c = 30 * np.log(D / delta)
        self.delta = delta
        self.D = D
        self.w_i = np.zeros((D,))
        self.q = q
        self.n_observed = 0

    def assign_next(self, x):
        c_prime = np.min([self.c, self.n_observed / self.q / 2])
        self.n_observed += 1
        # Line four
        if (np.abs(x @ self.w_i) > c_prime) or (np.max(self.w_i) > c_prime):
            p_i = self.q  # fallback to CR rather than failing
        else:
            p_i = self.q - self.q * ((x @ self.w_i) / (c_prime))
        assignment = 1 / 2 / self.q if np.random.rand() < p_i else -1 / 2 / (1 - self.q)
        self.w_i += assignment * x
        return int((assignment + 1) / 2)
