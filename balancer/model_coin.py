import numpy as np
from sklearn.linear_model import SGDClassifier

from .base import Balancer


class ModelCoin(Balancer):
    def __init__(self, q=0.5, **kwargs):
        self.sgd_clss = SGDClassifier(
            loss="log",  # class_weight={1: 1 / q, 0: 1 / (1 - q)},
            # learning_rate='constant', eta0=1e-5,
            # fit_intercept=False, tol=None, shuffle=False
            average=2,
        )
        self.q = q
        self.classes = np.array([0, 1])

    def assign_all(self, X):
        first = np.sign(np.random.randn()).item()
        self.sgd_clss.partial_fit(X[0, None], [first], classes=self.classes)
        return np.array([first] + [self.assign_next(x) for x in X[1:, None]])

    def assign_next(self, x):
        p_i = self.sgd_clss.predict_proba(x)[0, 0]
        # print(p_i)
        assignment = 1 if np.random.rand() < p_i else 0
        self.sgd_clss.partial_fit(x, [assignment])
        return assignment
