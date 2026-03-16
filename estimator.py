from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor

from balancer.base import Balancer


@dataclass(frozen=True)
class Estimate:
    estimate: float
    std_error: float


@dataclass(frozen=True)
class MultiTreatmentEstimate:
    treatment_group: str
    estimate: Estimate


class Estimator(metaclass=ABCMeta):
    def __init__(self, design: Balancer) -> None:
        self.design = design

    @abstractmethod
    def ATE(self, X, A, YA) -> Estimate:
        pass


class DifferenceInMeans(Estimator):
    def _diff_in_means(self, Y, A):
        if np.sum(A == 0) == 0 or np.sum(A == 0) == len(A):
            return np.inf
        return np.average(Y[A != 0]) - np.average(Y[A == 0])

    def _var_for_diff_in_means(self, Y, A):
        if np.sum(A) == 0 or np.sum(A) == len(A):
            return np.inf
        return np.var(Y[A != 0]) / np.sum(A != 0) + np.var(Y[A == 0]) / np.sum(A == 0)

    def ATE(self, X, A, YA) -> Estimate:
        return Estimate(
            estimate=self._diff_in_means(YA, A),
            std_error=np.sqrt(self._var_for_diff_in_means(YA, A)),
        )


class MultiEstimates(Estimator):
    def __init__(self, design: Balancer, base_estimator_class) -> None:
        self.est = base_estimator_class(design)
        super().__init__(design)

    def ATE(self, X, A, YA):
        ctl = 0
        return [
            MultiTreatmentEstimate(
                treatment_group=grp,
                estimate=self.est.ATE(
                    X[np.logical_or(ctl == A, grp == A), :],
                    A[np.logical_or(ctl == A, grp == A)],
                    YA[np.logical_or(ctl == A, grp == A)],
                ),
            )
            for grp in range(1, self.design.K + 1)
        ]


class MultiDiffInMeans(MultiEstimates):
    def __init__(self, design):
        super().__init__(design, DifferenceInMeans)


class MultiOLS(MultiEstimates):
    def __init__(self, design):
        super().__init__(design, CovariateAdjustedMean)


class Blocking(DifferenceInMeans):
    def ATE(self, X, A, YA) -> Estimate:
        cates = []
        ns = []
        vars = []
        overall_var = np.var(YA[A == 1]) + np.var(YA[A == 0])
        for block in self.design.blocks:
            ns.append(len(block))
            cates.append(self._diff_in_means(YA[block], A[block]))
            block_var = self._var_for_diff_in_means(YA[block], A[block])
            vars.append(block_var if block_var > 0 else overall_var)
        ns = np.array(ns) / np.sum(ns)
        return Estimate(
            estimate=np.average(cates, weights=ns),
            std_error=np.sqrt(np.average(vars, weights=np.power(ns, 2))),
        )


class KNNT(DifferenceInMeans):
    def ATE(self, X, A, YA) -> Estimate:
        int(np.sqrt(A.shape[0]))
        min_units = min(sum(A), sum(1 - A))
        k = 5 if min_units >= 5 else min_units
        knn0 = KNeighborsRegressor(
            weights="distance", n_neighbors=k, algorithm="kd_tree"
        )
        knn1 = KNeighborsRegressor(
            weights="distance", n_neighbors=k, algorithm="kd_tree"
        )
        knn0.fit(X[A == 0, :], YA[A == 0])
        knn1.fit(X[A == 1, :], YA[A == 1])
        y0 = knn0.predict(X)
        y1 = knn1.predict(X)
        return Estimate(estimate=np.average(y1 - y0), std_error=np.array([0]))


class MatchedPairBlocking(DifferenceInMeans):
    def ATE(self, X, A, YA) -> Estimate:
        cates = []
        for block in self.design.blocks:
            cates.append(self._diff_in_means(YA[block], A[block]))
        return Estimate(
            estimate=np.average(cates),
            std_error=np.sqrt(np.var(cates) / len(self.design.blocks)),
        )


class CovariateAdjustedMean(Estimator):
    def ATE(self, X, A, YA) -> Estimate:
        A = A.reshape(-1, 1)
        X = preprocessing.scale(X)
        XA = A * X
        X = sm.add_constant(np.hstack((A, X, XA)))
        model = sm.OLS(YA, X)
        results = model.fit(cov_type="HC0")
        return Estimate(estimate=results.params[1], std_error=results.bse[1])
