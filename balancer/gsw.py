# must have julia installed and GSWDesign installed for
# this to work
# https://github.com/crharshaw/GSWDesign.jl/
from julia.GSWDesign import sample_gs_walk

from .base import Balancer


class GSW(Balancer):
    def __init__(self, lamb=0.5, q=0.5, balanced=True):
        self.q = q
        self.lamb = lamb
        self.balanced

    def assign_all(self, X):
        sample_gs_walk(
            X, self.lamb, balanced=self.balanced, num_samples=1, treatment_probs=self.q
        )
