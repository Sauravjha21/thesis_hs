import numpy as np


class Balancer:
    def __init__(self):
        pass

    def assign_next(self, x):
        raise NotImplementedError("Balancer is abstract.")

    def assign_all(self, X):
        return np.array([self.assign_next(X[i, :]) for i in range(X.shape[0])])
