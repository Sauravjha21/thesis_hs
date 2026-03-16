#! /usr/bin/python3


import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree

from .base import Balancer


class QuickBlock(Balancer):
    def __init__(
        self,
        k: int = 1,
        q: float = 0.5,
        break_big_blocks: bool = True,
        order_seeds_by_indegree: bool = True,
        **kwargs,
    ):
        self.k = k
        self.q = q
        self.break_big_blocks = break_big_blocks
        self.order_seeds_by_indegree = order_seeds_by_indegree
        super().__init__()

    def _break_big_blocks(self, metric: str) -> None:
        block_copy = self.blocks.copy()
        for block_id, block_members in enumerate(block_copy):
            if len(block_members) <= (2 * (self.k + 1)):
                continue
            distances = pairwise_distances(self.Xt[block_members, :], metric=metric)
            max_dist = np.max(distances)
            loc_i, loc_j = np.where(distances == max_dist)
            seed_i = block_members[loc_i[0]]
            seed_j = block_members[loc_j[0]]
            members_i = self.knn[seed_i]
            members_j = [k for k in self.knn[seed_j] if k not in members_i]
            loners = [
                ind
                for ind, k in enumerate(block_members)
                if k not in members_i and k not in members_j
            ]
            loner_i_dist = distances[loners, loc_i[0]].reshape(-1)
            loner_j_dist = distances[loners, loc_j[0]].reshape(-1)
            for i_loner, loner in enumerate(loners):
                if loner_i_dist[i_loner] <= loner_j_dist[i_loner]:
                    members_i = np.append(members_i, block_members[loner])
                else:
                    members_j = np.append(members_j, block_members[loner])
            if len(members_j) == 1:
                neighbor = self.knn[seed_j][1]
                members_j = np.array([seed_j, neighbor])
                members_i = [m for m in members_i if m != neighbor]
            self.blocks[block_id] = members_i
            for member in members_i:
                self.block_membership[member] = block_id
            self.blocks.append(members_j)
            for member in members_j:
                self.block_membership[member] = len(self.blocks) - 1

    def _fit(self, X: np.ndarray, distance="mahalanobis") -> None:
        self.X = X
        if distance == "mahalanobis" and X.shape[1] > 1:
            self.transformer = np.linalg.cholesky(
                np.linalg.pinv(np.cov(X, rowvar=False))
            )
            self.Xt = self.X @ self.transformer
        #  Mahalanobis if d=1
        elif distance == "mahalanobis" and X.shape[1] == 1:
            self.Xt = self.X
            self.transformer = np.array([1 / np.var(X)])
        elif distance == "euclidean":
            self.Xt = self.X
            if self.X.shape[1] > 1:
                self.transformer = np.eye(self.X.shape[1])
            else:
                self.transformer = np.array([1])
        else:
            raise NotImplementedError(
                "Only Mahalanobis and Euclidean distance are implemented."
            )
        # maybe permute X first, here?
        N = self.Xt.shape[0]
        # step 1
        self.tree = KDTree(self.Xt, leaf_size=2, metric="euclidean")
        self.dist, self.knn = self.tree.query(self.Xt, k=self.k + 1)

        if self.order_seeds_by_indegree:
            indegree = [0] * N
            for i in range(N):
                neighbors = self.knn[i][1:]
                for neighbor in neighbors:
                    indegree[neighbor] += 1
            seed_order = np.argsort(indegree)
        else:
            seed_order = range(N)

        # step 2
        seeds: list[int] = []
        adjacents: set[int] = set()

        def in_neighborhood(neighbors, adjacents):
            return any(neighbor in adjacents for neighbor in neighbors)

        for i in seed_order:
            if i in adjacents:
                continue
            neighborhood = self.knn[i]
            if in_neighborhood(neighborhood[1:], adjacents):
                continue

            seeds.append(i)
            adjacents.update(neighborhood)

        # step 3
        self.blocks: list[list[int]] = []
        self.block_membership = np.array([-1] * N)
        for block, seed in enumerate(seeds):
            neighborhood = self.knn[seed]
            self.blocks.append(list(neighborhood))
            for member in neighborhood:
                self.block_membership[member] = block

        # step 4
        for i in range(N):
            if self.block_membership[i] >= 0:
                continue
            closest_unit = self.knn[i][1]
            # print(i, closest_unit)
            self.block_membership[i] = self.block_membership[closest_unit]
            self.blocks[self.block_membership[i]].append(i)

        # break up big blocks
        if self.break_big_blocks:
            self._break_big_blocks("euclidean")
        self.blocks = [list(block) for block in self.blocks]

    def assign_all(self, X: np.ndarray) -> np.ndarray:
        self._fit(X)

        N = X.shape[0]

        A = np.array([0] * N)
        n_trt = np.zeros((len(self.blocks),), dtype=np.int8)
        remainder = np.zeros((len(self.blocks),))
        for idx, block in enumerate(self.blocks):
            M = len(block)
            n_trt[idx] = int(max(1, np.floor(M * self.q)))
            remainder[idx] = (
                n_trt[idx] - M * self.q + np.random.uniform(low=0.0, high=1e-5)
            )

        N_extra = int(max(1, np.floor(N * self.q)) - sum(n_trt))
        if N_extra > 0:
            for idx in remainder.argsort()[:N_extra]:
                n_trt[idx] = n_trt[idx] + 1

        for idx, block in enumerate(self.blocks):
            M = len(block)
            trted = np.random.choice(M, n_trt[idx], replace=False)
            for unit in trted:
                A[block[unit]] = 1
        return A
