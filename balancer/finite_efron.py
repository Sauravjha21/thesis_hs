import numpy as np

from .base import Balancer

c_dict = {
    "alweiss": lambda kw: 30 * np.log(kw["N"] * kw["D"] / kw["delta"]),
    "sqrt_of_var": lambda kw: 8
    * np.log(kw["D"] / kw["delta"])
    / np.sqrt(kw["q"] * (1 - kw["q"])),
    "squared_q": lambda kw: 8 * np.log(kw["D"] / kw["delta"]) / (kw["q"] ** 2),
}

dynamic_c_dict = {
    "none": lambda kw: kw["c"],
    "linear": lambda kw: np.min([kw["c"], kw["n_observed"] / kw["q"] / 2]),
}


class FiniteEfron(Balancer):
    def __init__(
        self,
        N,
        D,
        delta=0.05,
        q=0.5,
        c_method="squared_q",
        dynamic_c="linear",
        **kwargs,
    ):
        """
        Args:
            N (int): total number of points
            D (int): dimension of the data
            delta (float): probability of success
        """
        # print(c_method, dynamic_c)
        self.n_observed = 0
        self.c = c_dict.get(c_method, 1.0)({"N": N, "D": D, "delta": delta, "q": q})
        self.dynamic_c = dynamic_c
        self.delta = delta
        self.D = 1
        self.w_i = np.zeros((D,))
        self.q = q if q < 0.5 else 1 - q
        self.mu = np.zeros((D,))
        self.Sigma = np.zeros((D, D))
        self.value_plus = 1 / (2 * self.q)
        self.value_minus = -1 / (2 * (1 - self.q))

    def assign_next(self, x):
        self.n_observed += 1
        c_prime = dynamic_c_dict[self.dynamic_c](
            {"c": self.c, "n_observed": self.n_observed, "q": self.q}
        )

        if any(self.w_i > c_prime):
            p_i = self.q
        else:
            dot = np.dot(x, self.w_i).item()
            if np.abs(dot) > c_prime:
                p_i = 1 * (np.abs(dot) > c_prime)
            else:
                p_i = self.q * (1 - dot / c_prime)

        if np.random.rand() < p_i:
            assignment = 1
            self.w_i += self.value_plus
        else:
            assignment = 0
            self.w_i += self.value_minus
        return assignment
