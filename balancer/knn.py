import numpy as np

from .base import Balancer


class NeighborCoin(Balancer):
    def __init__(self, q=0.5, correction=10, neighbors=3, **kwargs):
        self.observations = []
        self.labels = []
        self.neighbors = neighbors
        #        self.get_classifier = lambda n_samples: knn_classifier(n_neighbors=int(np.sqrt(n_samples)), weights="distance", metric='manhattan') if n_samples > neighbors else knn_classifier(n_neighbors=n_samples, weights="distance", metric='manhattan')
        self.index = None
        q = q if q >= 0.5 else 1 - q
        odds = q / (1 - q)
        self.q = q
        self.count = 0
        self.q_plus = (odds * correction) / (odds * correction + 1)
        self.q_minus = (odds / correction) / (odds / correction + 1)

    def assign_all(self, X):
        first = int((np.sign(np.random.randn()).item() + 1) / 2)
        # self.observations.append(X[0, :])
        self.labels.append(first)
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(np.atleast_2d(X[0, :]).astype(np.float32))
        self.count = 1
        for idx in range(X.shape[0] - 1):
            self.assign_next(X[idx + 1, :])
        return np.array(self.labels)

    def assign_next(self, x):
        if self.index is None:
            self.index = faiss.IndexFlatL2(x.shape[0])  # build the index
        # classifier = self.get_classifier(len(self.observations))
        # classifier.fit(np.array(self.observations), np.array(self.labels))

        D, I = self.index.search(
            np.atleast_2d(x).astype(np.float32), int(np.sqrt(self.count))
        )
        p_i = 1 - (np.array(self.labels)[I[0]]).mean()
        diff = self.q - p_i
        if diff == 0:
            pr = self.q
        elif diff > 0:
            pr = self.q_minus
        else:
            pr = self.q_plus

        assignment = 1 if np.random.rand() < pr else 0

        # self.observations.append(x)
        self.labels.append(assignment)
        self.index.add(np.atleast_2d(x).astype(np.float32))
        self.count += 1
        return assignment
