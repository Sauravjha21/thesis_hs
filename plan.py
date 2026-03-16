#! /usr/bin/python3

import time
from abc import ABCMeta

import numpy as np
import pandas as pd
from sklearn import preprocessing

from balancer import Balancer
from estimator import Estimator
from evaluator import Evaluator


class Plan(metaclass=ABCMeta):
    def __init__(self):
        self.evaluators = {}
        self.designs = {}

    def add_design(
        self,
        design_name,
        design_class: type[Balancer],
        estimator_class: type[Estimator],
        design_kwargs=None,
    ):
        if design_kwargs is None:
            design_kwargs = {}
        self.designs[design_name] = (design_class, estimator_class, design_kwargs)

    def add_evaluator(
        self, evaluator_name: str, evaluator_class: type[Evaluator], eval_kwargs=None
    ):
        if eval_kwargs is None:
            eval_kwargs = {}
        self.evaluators[evaluator_name] = (evaluator_class, eval_kwargs)

    def execute(self, dgp_factory, seed):
        np.random.seed(seed)
        dgp = dgp_factory.create_dgp()
        X = dgp.X
        # X = preprocessing.scale(X) / dgp.D
        # print(
        #     f"""
        #     avgL2: {np.power(X, 2).sum(axis = 1).mean()},\t
        #     maxL2: {np.power(X, 2).sum(axis = 1).max()},\t
        #     num>1: {np.sum(np.power(X, 2).sum(axis = 1) > 1)},\t
        #     qs: {np.quantile(np.power(X, 2).sum(axis = 1), [0.75, 0.9, 0.95, 0.975, 0.99, 0.995])}
        #     """
        # )
        X = preprocessing.scale(X)
        # X = X / np.power(X, 2).sum(axis = 1).max()
        X = X / np.sqrt(np.power(X, 2).sum(axis=1).max())
        all_y = dgp.potential_outcomes
        results = []
        for design_name, (
            design_class,
            estimator_class,
            design_kwargs,
        ) in self.designs.items():
            # print(design_name)
            def make_row(name, value):
                return pd.DataFrame(
                    {"design": [design_name], "metric": [name], "value": [value]}
                )

            time_start = time.time()
            design_kwargs["N"] = dgp.n
            design_kwargs["D"] = dgp.D
            design = design_class(**design_kwargs)
            A = design.assign_all(X)
            time_end = time.time()
            time_elapsed = time_end - time_start
            results.append(make_row("time_design", time_elapsed))
            YA = dgp.Y(A)
            time_start = time.time()
            estimator = estimator_class(design)
            ATEhat = estimator.ATE(X, A, YA)
            time_end = time.time()
            time_elapsed = time_end - time_start
            results.append(make_row("time_estimation", time_elapsed))
            for name, (evaluator_class, eval_kwargs) in self.evaluators.items():
                evaluator = evaluator_class(dgp, **eval_kwargs)
                val = evaluator.evaluate(X, all_y, A, YA, ATEhat)
                results.append(make_row(name, val))
        return pd.concat(results)
