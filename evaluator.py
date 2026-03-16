"""
evaluator.py
============
Evaluation metrics for experimental design simulations.

Every evaluator receives the full information from one completed experiment
and returns a single scalar summarising some aspect of performance. Evaluators
are registered with a Plan and called automatically after each design run.

Shared evaluate() signature
----------------------------
All evaluators implement:

    evaluate(X, all_Y, A, YA, ATEhat) -> Number

Parameters
----------
X : np.ndarray, shape (n, d)
    Scaled covariate matrix. Rows are subjects, columns are covariates.
    Scaling (standardise then max-L2-normalise) is applied by plan.py
    before passing X to both the balancer and the evaluators.
all_Y : np.ndarray, shape (n, 2)
    Full potential outcomes table. Column 0 is Y_i(0) (control outcome),
    column 1 is Y_i(1) (treatment outcome). In a real experiment only one
    column per row is observed; both are available here because this is a
    simulation where the ground truth is known.
A : np.ndarray, shape (n,)
    Assignment vector produced by the balancer. Values are 0 (control)
    or 1 (treatment) for the binary case, or 0..K for multi-treatment.
YA : np.ndarray, shape (n,)
    Observed outcomes — i.e. all_Y[i, A[i]] for each subject i.
ATEhat : Estimate
    Object with two fields:
      .estimate  (float) — point estimate of the average treatment effect
      .std_error (float) — Neyman standard error of the estimate

Contents
--------
Standard metrics (from original codebase, ATECovers bug fixed):
    Evaluator               — abstract base class
    AvgMarginalProb         — average marginal treatment probability
    EntropyMarginalProb     — entropy of the marginal treatment distribution
    ATEError                — signed error: true ATE minus estimated ATE
    CovariateMSE            — mean squared covariate imbalance between groups
    ATECovers               — Neyman CI coverage (bug fixed)
    CISize                  — width of the Neyman 95% CI
    MultiTreatmentEvaluator — adapter for K > 1 treatment arms

Neyman power baseline (new):
    NeymanRejects           — 1 if Neyman t-test rejects H0 at level alpha

Randomisation inference (new, thesis contribution):
    RandomisationInferencePValue    — p-value from RI null distribution
    RandomisationInferenceRejects   — 1 if RI p-value < alpha (power)
    RandomisationInferenceCoverage  — 1 if true ATE inside RI CI (coverage)

Helper functions (module-private, prefixed _):
    _diff_in_means    — raw difference-in-means estimate
    _neyman_se        — Neyman standard error
    _t_stat           — difference-in-means t-statistic
    _null_assignments — re-run a balancer B times, return (n, B) assignments
"""

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import norm

NORMAL_QUANTILE = norm.ppf(0.975)


class Evaluator(metaclass=ABCMeta):
    """
    Abstract base class for all evaluators.

    An evaluator takes the complete output of one experiment run and returns
    a single scalar measuring some aspect of performance (bias, coverage,
    power, imbalance, etc.).

    Subclasses must implement the evaluate() method. The __init__ receives
    the DGP object so that evaluators which need ground-truth information
    (e.g. the true ATE, or the balancer parameters) can store it at
    construction time.

    Parameters
    ----------
    dgp : DGP
        The data-generating process for the current simulation iteration.
        Provides access to ground-truth quantities such as potential outcomes
        and true sample size.
    """

    def __init__(self, dgp) -> None:
        self.dgp = dgp

    @abstractmethod
    def evaluate(self, X, all_Y, A, YA, ATEhat) -> float:
        """
        Evaluate one completed experiment and return a scalar metric.

        See module docstring for the full description of each argument.
        """
        pass


# ── Existing evaluators (ATECovers bug fixed) ─────────────────────────────────


class AvgMarginalProb(Evaluator):
    """
    Average marginal probability of treatment assignment.

    Computes the mean fraction of subjects assigned to each non-control
    treatment arm. For the binary case (K=1) this is simply the fraction
    of subjects assigned to treatment. For K > 1 arms it averages the
    per-arm fractions.

    A perfectly balanced design returns 0.5. Values far from 0.5 indicate
    that the balancer produced unequal group sizes, which reduces efficiency.

    Returns
    -------
    float
        Average marginal treatment probability across all non-control arms.
        Returns 0.0 if no subjects were assigned to treatment.
    """

    def evaluate(self, X, all_Y, A, YA, ATEhat) -> float:
        return (
            np.average([np.average(a == A) for a in np.unique(A) if a != 0])
            if np.sum(A) > 0
            else 0.0
        )


class EntropyMarginalProb(Evaluator):
    """
    Normalised entropy of the marginal treatment assignment distribution.

    Measures how evenly subjects are spread across treatment arms. Entropy
    is normalised by log(K) so that the maximum value is always 1.0,
    regardless of the number of arms K.

    Interpretation:
      1.0 — perfectly uniform split across all arms (ideal balance)
      0.0 — all subjects assigned to a single arm (complete imbalance)

    This complements AvgMarginalProb: two designs can have the same average
    marginal probability but different entropy if the distribution across
    arms is uneven.

    Returns
    -------
    float
        Normalised entropy in [0, 1]. Returns 0 if only one unique
        assignment value is observed (degenerate case).
    """

    def evaluate(self, X, all_Y, A, YA, ATEhat) -> float:
        pr = [np.average(a == A) for a in np.unique(A)]
        if len(pr) > 1:
            sumand = [p * np.log(p) / np.log(len(pr)) for p in pr if p > 0]
            return -np.sum(sumand)
        else:
            return 0


class ATEError(Evaluator):
    """
    Signed estimation error: true ATE minus estimated ATE.

    Measures the bias of the point estimator in a single experiment run.
    Averaging this over many iterations gives the empirical bias of the
    estimator under the given design.

        ATEError = tau - tau_hat

    where tau = mean(Y_i(1) - Y_i(0)) is the true average treatment effect
    computed from the full potential outcomes table, and tau_hat is the
    difference-in-means estimate from the observed data.

    A value close to zero on average indicates an unbiased estimator.
    Positive values mean the estimator underestimates the true effect;
    negative values mean it overestimates.

    Returns
    -------
    float
        True ATE minus estimated ATE. Can be negative.
    """

    def evaluate(self, X, all_Y, A, YA, ATEhat) -> float:
        ATE = np.average(all_Y[:, 1] - all_Y[:, 0])
        return ATE - ATEhat.estimate


class CovariateMSE(Evaluator):
    """
    Mean squared covariate imbalance between treatment and control groups.

    Computes the average squared difference between the treatment-group mean
    and the control-group mean for each covariate, then averages across
    covariates:

        CovariateMSE = (1/d) * sum_k (mean(X_k | treated) - mean(X_k | control))^2

    This directly measures how well the assignment balanced the two groups
    on their background characteristics. A value of 0 means perfect balance;
    larger values indicate greater imbalance.

    Under pure randomisation, CovariateMSE decreases at rate O(1/n). Under
    BWD, it decreases much faster — this is the central empirical claim of
    the BWD paper (Table 1).

    Returns
    -------
    float
        Mean squared covariate imbalance across all d covariates. Always >= 0.
    """

    def evaluate(self, X, all_Y, A, YA, ATEhat) -> float:
        X1 = np.average(X[A != 0, :], 0)
        X0 = np.average(X[A == 0, :], 0)
        return np.mean(np.power(X1 - X0, 2)).item()


class ATECovers(Evaluator):
    """
    Neyman 95% confidence interval coverage indicator.

    Returns 1 if the true ATE falls inside the Neyman-based 95% CI, 0
    otherwise. Averaging over many iterations gives the empirical coverage
    rate, which should be at or above 0.95 for a valid inferential procedure.

    The CI is constructed as:
        [tau_hat - 1.96 * SE_Neyman,  tau_hat + 1.96 * SE_Neyman]

    where SE_Neyman = sqrt(var(Y|T=1)/n_1 + var(Y|T=0)/n_0) is the
    conservative Neyman standard error that ignores the experimental design.

    Because Neyman SE is conservative under balanced designs like BWD, this
    CI tends to be wider than necessary, leading to over-coverage (empirical
    coverage > 0.95). This motivates the randomisation inference approach
    which produces tighter, more honest CIs.

    Bug fix
    -------
    The original codebase computed `all_Y[:,1] - all_Y[:,1]` (always zero)
    instead of `all_Y[:,1] - all_Y[:,0]`. This meant the evaluator was
    checking whether 0 is inside the CI rather than whether the true ATE is,
    making the coverage metric completely meaningless. This has been fixed.

    Returns
    -------
    bool
        True if the true ATE is inside the Neyman 95% CI, False otherwise.
        Returns False if the estimate is infinite (degenerate assignment).
    """

    def evaluate(self, X, all_Y, A, YA, ATEhat) -> float:
        ATE = np.average(all_Y[:, 1] - all_Y[:, 0])  # fixed: was [:, 1] - [:, 1]
        if ATEhat.estimate == np.inf:
            return False
        lwr = ATEhat.estimate - NORMAL_QUANTILE * ATEhat.std_error
        upr = ATEhat.estimate + NORMAL_QUANTILE * ATEhat.std_error
        return bool((lwr <= ATE) & (upr >= ATE))


class CISize(Evaluator):
    """
    Width of the Neyman 95% confidence interval.

    Measures the precision of the estimator. Smaller CI widths indicate
    more precise inference. Under BWD, the Neyman CI is wider than necessary
    because the SE is conservative — it does not exploit the design's
    variance reduction. Comparing CISize against the RI CI width (derived
    from RandomisationInferenceCoverage) quantifies how much precision is
    being wasted by ignoring the design.

        CISize = 2 * 1.96 * SE_Neyman

    Returns
    -------
    float
        Total width of the 95% CI. Always >= 0.
    """

    def evaluate(self, X, all_Y, A, YA, ATEhat) -> float:
        return 2 * NORMAL_QUANTILE * ATEhat.std_error


class MultiTreatmentEvaluator(Evaluator):
    """
    Adapter that extends any binary evaluator to handle K > 1 treatment arms.

    For experiments with multiple treatment groups (K arms plus one control),
    this class applies a base evaluator to each treatment-vs-control comparison
    separately and returns the average result across all K comparisons.

    For each treatment arm k in {1, ..., K}:
      1. Select only subjects in control (A=0) or arm k (A=k).
      2. Extract the corresponding slice of potential outcomes.
      3. Apply the base evaluator to this binary sub-experiment.
      4. Average the K results.

    Parameters
    ----------
    dgp : DGP
        The data-generating process. Must have dgp.K >= 1.
    base_eval_class : type
        A binary Evaluator subclass (e.g. ATEError, ATECovers) to apply
        to each treatment-vs-control comparison. Required — raises ValueError
        if not provided.

    Raises
    ------
    ValueError
        If base_eval_class is None, or if the number of estimates in ATEhat
        does not match dgp.K.

    Returns
    -------
    float
        Average of the base evaluator's output across all K treatment arms.
    """

    def __init__(self, dgp, base_eval_class=None):
        if base_eval_class is None:
            raise ValueError("Need a base class for Multi-treatment evaluation.")
        self.base = base_eval_class(dgp)
        self.dgp = dgp

    def evaluate(self, X, all_Y, A, YA, ATEhat):
        ctl = 0
        result = 0.0
        if len(ATEhat) != self.dgp.K:
            raise ValueError("number of estimates is different than number of groups")
        for i in range(self.dgp.K):
            trt = i + 1
            ok = np.logical_or(ctl == A, trt == A)
            this_y = np.vstack((all_Y[ok, ctl], all_Y[ok, trt])).T
            result += self.base.evaluate(
                X[ok, :], this_y, A[ok], YA[ok], ATEhat[i].estimate
            )
        return result / self.dgp.K


# ── Neyman power baseline ─────────────────────────────────────────────────────


class NeymanRejects(Evaluator):
    """
    Power indicator: returns 1 if the Neyman t-test rejects H0: tau=0.

    This is the standard frequentist power baseline. It tests whether the
    difference-in-means estimate is significantly different from zero using
    the Neyman standard error, and returns 1 if the two-sided p-value falls
    below alpha.

        t = tau_hat / SE_Neyman
        p = 2 * (1 - Phi(|t|))
        NeymanRejects = 1{p < alpha}

    Averaging NeymanRejects over many simulation iterations gives the
    empirical power of the Neyman test — the fraction of times we correctly
    detect a true treatment effect. This is compared against RI_Rejects
    (RandomisationInferenceRejects) to quantify the power gain from using
    randomisation inference instead.

    Parameters
    ----------
    dgp   : DGP
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    int
        1 if the Neyman t-test rejects H0 at level alpha, 0 otherwise.
        Returns 0 for degenerate cases (SE = 0 or infinite).
    """

    def __init__(self, dgp, alpha=0.05):
        super().__init__(dgp)
        self.alpha = alpha

    def evaluate(self, X, all_Y, A, YA, ATEhat) -> float:
        if ATEhat.std_error == 0 or ATEhat.std_error == np.inf:
            return 0
        t_stat = ATEhat.estimate / ATEhat.std_error
        p_value = 2 * (1 - norm.cdf(np.abs(t_stat)))
        return int(p_value < self.alpha)


# ══════════════════════════════════════════════════════════════════════════════
# Randomisation Inference
# ══════════════════════════════════════════════════════════════════════════════
#
# Theoretical justification:
#   BWD's assignment A_i depends only on (x_1, ..., x_{i-1}) and the running
#   imbalance w_{i-1} — never on outcomes. So the assignment mechanism is fully
#   known and reproducible given the same covariate sequence X.
#
#   Under the sharp null H0: Y_i(1) = Y_i(0) for all i, observed outcomes YA
#   are fixed regardless of assignment. Re-running BWD on the same X and
#   evaluating the test statistic with the real YA gives a valid null
#   distribution for inference.
#
# Implementation notes:
#   - X arrives at evaluators already scaled by plan.py (standardised + max
#     L2-normalised). We pass it directly to the re-runs — no re-scaling needed.
#   - We always copy balancer_kwargs before injecting N and D to avoid mutating
#     the shared dict (plan.py has the same mutation pattern on line 54).
#   - _null_assignments() generates all B assignment vectors up front so the
#     three RI evaluators can each reuse the same helper function cleanly.
#
# ══════════════════════════════════════════════════════════════════════════════


def _diff_in_means(YA: np.ndarray, A: np.ndarray) -> float:
    """
    Compute the raw difference-in-means estimate.

    Returns mean(Y | treated) - mean(Y | control). Returns 0.0 if either
    group is empty (degenerate assignment).

    Parameters
    ----------
    YA : np.ndarray, shape (n,)  — observed outcomes
    A  : np.ndarray, shape (n,)  — assignment vector (0=control, 1=treatment)

    Returns
    -------
    float
    """
    n_t = np.sum(A != 0)
    n_c = np.sum(A == 0)
    if n_t == 0 or n_c == 0:
        return 0.0
    return float(np.mean(YA[A != 0]) - np.mean(YA[A == 0]))


def _neyman_se(YA: np.ndarray, A: np.ndarray) -> float:
    """
    Compute the Neyman (conservative) standard error for the
    difference-in-means estimator.

        SE = sqrt(var(Y|T=1) / n_1  +  var(Y|T=0) / n_0)

    This is conservative because it does not exploit any covariance reduction
    from the experimental design. Under BWD, the true variance of tau_hat is
    smaller than SE^2 because the design actively balances covariates.

    Returns np.inf if either group is empty.

    Parameters
    ----------
    YA : np.ndarray, shape (n,)
    A  : np.ndarray, shape (n,)

    Returns
    -------
    float
    """
    n_t = np.sum(A != 0)
    n_c = np.sum(A == 0)
    if n_t == 0 or n_c == 0:
        return np.inf
    return float(np.sqrt(np.var(YA[A != 0]) / n_t + np.var(YA[A == 0]) / n_c))


def _t_stat(YA: np.ndarray, A: np.ndarray) -> float:
    """
    Compute the Neyman t-statistic: difference-in-means divided by Neyman SE.

    Used as the test statistic throughout the randomisation inference
    procedure. Returns 0.0 for degenerate cases (empty group or zero SE).

    Parameters
    ----------
    YA : np.ndarray, shape (n,)
    A  : np.ndarray, shape (n,)

    Returns
    -------
    float
    """
    se = _neyman_se(YA, A)
    if se == 0 or se == np.inf:
        return 0.0
    return _diff_in_means(YA, A) / se


def _null_assignments(
    X: np.ndarray, balancer_class, balancer_kwargs: dict, B: int
) -> np.ndarray:
    """
    Re-run a balancer B times on the same covariate matrix X and collect
    all assignment vectors into a single matrix.

    This is the core computational step for randomisation inference. By
    re-running the balancer with different random seeds (the balancer's
    internal randomness determines the seed), we build an empirical
    distribution of what assignments could have been observed under the
    same experimental conditions.

    X is assumed to be already scaled exactly as plan.py scales it before
    calling the balancer in the real experiment — so no re-scaling is done
    here. N and D are read from X.shape and injected into balancer_kwargs
    for each run. The original balancer_kwargs dict is never mutated.

    Parameters
    ----------
    X               : np.ndarray, shape (n, d) — pre-scaled covariate matrix
    balancer_class  : class — balancer to re-run (e.g. bal.BWD)
    balancer_kwargs : dict  — kwargs for the balancer excluding N and D
    B               : int   — number of re-runs

    Returns
    -------
    np.ndarray, shape (n, B)
        Column b contains the assignment vector from re-run b.
        Values are 0 (control) or 1 (treatment).
    """
    N, D = X.shape
    assignments = np.zeros((N, B), dtype=int)
    for b in range(B):
        kwargs = dict(balancer_kwargs)  # copy — never mutate the original
        kwargs["N"] = N
        kwargs["D"] = D
        assignments[:, b] = balancer_class(**kwargs).assign_all(X)
    return assignments


class RandomisationInferencePValue(Evaluator):
    """
    Two-sided p-value for the sharp null H0: Y_i(1) = Y_i(0) for all i.

    Theoretical basis
    -----------------
    BWD's assignment rule A_i depends only on (x_1, ..., x_{i-1}) and the
    running imbalance w_{i-1} — never on outcomes. This means the assignment
    mechanism is fully known and can be replicated exactly given the same
    covariate sequence. Under the sharp null, potential outcomes are fixed
    regardless of assignment, so re-running BWD on the same X and computing
    the test statistic with the real outcomes YA gives a valid draw from the
    null distribution of the test statistic.

    Procedure
    ---------
    1. Compute the real t-statistic t* = _t_stat(YA, A) from the actual
       experiment.
    2. Re-run the balancer B times on the same (already-scaled) X to get
       B alternative assignment vectors A_1, ..., A_B.
    3. For each re-run b, compute t_b = _t_stat(Y_null, A_b) where
       Y_null = all_Y[:, 0] is the control potential outcome for every subject.
       Under H0: Y_i(1) = Y_i(0), everyone's outcome equals their control
       outcome regardless of assignment — so Y_null is the correct fixed
       outcome vector to use.
    4. p-value = (1/B) * sum_b 1{|t_b| >= |t*|}.

    A small p-value means the real t-statistic is unlikely under the null,
    i.e. there is evidence of a treatment effect.

    Parameters
    ----------
    dgp             : DGP
    balancer_class  : class
        The balancer used in the real experiment (e.g. bal.BWD). Required.
    balancer_kwargs : dict
        Keyword arguments for the balancer excluding N and D
        (e.g. {'phi': 1.0, 'delta': 0.05}). N and D are injected
        automatically from X.shape at evaluation time.
    B               : int
        Number of re-runs to build the null distribution (default 500).
        B=100 is sufficient for exploratory runs; use B=500 or B=1000
        for final results to reduce Monte Carlo error in the p-value.

    Returns
    -------
    float
        Two-sided p-value in [0, 1].
    """

    def __init__(self, dgp, balancer_class=None, balancer_kwargs=None, B=500):
        super().__init__(dgp)
        if balancer_class is None:
            raise ValueError("balancer_class is required.")
        self.balancer_class = balancer_class
        self.balancer_kwargs = balancer_kwargs or {}
        self.B = B

    def evaluate(self, X, all_Y, A, YA, ATEhat) -> float:
        t_star = _t_stat(YA, A)
        # Under the sharp null H0: Y_i(1) = Y_i(0), each subject's outcome is
        # fixed at Y_i(0) regardless of assignment. We use all_Y[:, 0] as the
        # null outcome vector — what every subject would have produced had no
        # one received treatment.
        Y_null = all_Y[:, 0]
        null_A = _null_assignments(X, self.balancer_class, self.balancer_kwargs, self.B)
        null_t = np.array([_t_stat(Y_null, null_A[:, b]) for b in range(self.B)])
        return float(np.mean(np.abs(null_t) >= np.abs(t_star)))


class RandomisationInferenceRejects(Evaluator):
    """
    Power indicator: returns 1 if the RI p-value is below alpha.

    This is the randomisation inference counterpart of NeymanRejects.
    Averaging over many simulation iterations gives the empirical power of
    the RI test — the fraction of times it correctly detects a true effect.

    Why RI typically has higher power than Neyman under BWD
    -------------------------------------------------------
    BWD produces assignments that are more balanced than pure randomisation.
    As a result, the null distribution of the t-statistic under RI is more
    concentrated around zero (the BWD re-runs also produce balanced
    assignments). The real t-statistic from a true effect is therefore more
    extreme relative to this tighter null distribution, making the test more
    powerful. The Neyman approach cannot exploit this because it uses a
    normal approximation based on the conservative Neyman SE which ignores
    the design entirely.

    Parameters
    ----------
    dgp             : DGP
    balancer_class  : class — balancer to re-run (e.g. bal.BWD)
    balancer_kwargs : dict  — balancer kwargs excluding N and D
    B               : int   — number of re-runs (default 500)
    alpha           : float — significance level (default 0.05)

    Returns
    -------
    int
        1 if RI p-value < alpha (reject H0), 0 otherwise.
    """

    def __init__(
        self, dgp, balancer_class=None, balancer_kwargs=None, B=500, alpha=0.05
    ):
        super().__init__(dgp)
        if balancer_class is None:
            raise ValueError("balancer_class is required.")
        self.balancer_class = balancer_class
        self.balancer_kwargs = balancer_kwargs or {}
        self.B = B
        self.alpha = alpha

    def evaluate(self, X, all_Y, A, YA, ATEhat) -> float:
        t_star = _t_stat(YA, A)
        Y_null = all_Y[:, 0]
        null_A = _null_assignments(X, self.balancer_class, self.balancer_kwargs, self.B)
        null_t = np.array([_t_stat(Y_null, null_A[:, b]) for b in range(self.B)])
        p_value = float(np.mean(np.abs(null_t) >= np.abs(t_star)))
        return int(p_value < self.alpha)


class RandomisationInferenceCoverage(Evaluator):
    """
    Coverage indicator: returns 1 if the true ATE is inside the RI CI.

    Constructs a (1 - alpha) confidence interval using the empirical
    distribution of the difference-in-means estimator under randomisation,
    then checks whether the true ATE falls inside it.

    CI construction (t-inversion method)
    -------------------------------------
    Under H0: Y_i(1) = Y_i(0), every subject's outcome is fixed at Y_i(0).
    We compute the null t-statistic for B re-runs of the balancer using this
    fixed outcome vector. The alpha/2 and 1-alpha/2 quantiles of the null
    t-distribution give critical values c_lo and c_hi. We then invert these
    back to the effect scale using the real Neyman SE:

        lwr = tau_hat - c_hi * se_hat
        upr = tau_hat - c_lo * se_hat

    This is valid because under the null the t-statistic is approximately
    pivotal — its distribution does not depend on the unknown true effect.
    Inverting it gives a CI that accounts for the variance reduction from
    the design, unlike the Neyman CI which uses a normal approximation.

    Relationship to ATECovers
    -------------------------
    Both ATECovers and RandomisationInferenceCoverage return 1 if the true
    ATE is inside a 95% CI. They differ in how the CI is constructed:
      - ATECovers: normal CI using Neyman SE (conservative, wider)
      - RandomisationInferenceCoverage: RI CI using empirical null
        distribution (tighter, design-aware)

    Averaging both over many iterations and comparing gives the trade-off:
    Neyman tends to over-cover (e.g. 98%), RI should be close to 95%.

    Parameters
    ----------
    dgp             : DGP
    balancer_class  : class — balancer to re-run (e.g. bal.BWD)
    balancer_kwargs : dict  — balancer kwargs excluding N and D
    B               : int   — number of re-runs (default 500)
    alpha           : float — CI level is (1 - alpha) (default 0.05 → 95% CI)

    Returns
    -------
    int
        1 if the true ATE is inside the RI (1-alpha) CI, 0 otherwise.
    """

    def __init__(
        self, dgp, balancer_class=None, balancer_kwargs=None, B=500, alpha=0.05
    ):
        super().__init__(dgp)
        if balancer_class is None:
            raise ValueError("balancer_class is required.")
        self.balancer_class = balancer_class
        self.balancer_kwargs = balancer_kwargs or {}
        self.B = B
        self.alpha = alpha

    def evaluate(self, X, all_Y, A, YA, ATEhat) -> float:
        true_ATE = float(np.mean(all_Y[:, 1] - all_Y[:, 0]))
        tau_hat = ATEhat.estimate
        se_hat = _neyman_se(YA, A)

        # Build null distribution of the t-statistic under H0: Y_i(1) = Y_i(0).
        # Under the sharp null every subject's outcome is Y_i(0), so we evaluate
        # the test statistic on all_Y[:, 0] with each simulated assignment.
        Y_null = all_Y[:, 0]
        null_A = _null_assignments(X, self.balancer_class, self.balancer_kwargs, self.B)
        null_t = np.array([_t_stat(Y_null, null_A[:, b]) for b in range(self.B)])

        # Invert the null t-distribution to get CI bounds on the effect scale.
        # The alpha/2 and 1-alpha/2 quantiles of the null t give critical values
        # c_lo and c_hi. The CI is: [tau_hat - c_hi * se_hat, tau_hat - c_lo * se_hat]
        c_lo = float(np.quantile(null_t, self.alpha / 2))
        c_hi = float(np.quantile(null_t, 1 - self.alpha / 2))
        lwr = tau_hat - c_hi * se_hat
        upr = tau_hat - c_lo * se_hat

        return int((true_ATE >= lwr) & (true_ATE <= upr))
