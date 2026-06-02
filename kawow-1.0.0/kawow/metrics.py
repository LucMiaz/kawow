"""kawow.metrics
================
Additional evaluation metrics for partition coefficient models.

Functions
---------
jeffreys_bf_corr(r, n)                    → dict  BF₁₀ for H₁: ρ ≠ 0 (Ly et al. 2015)
compare_correlations_bf(r12, r13, r23, n) → dict  Pairwise comparison BF (Steiger 1980)
lin_ccc(y_true, y_pred)                   → float  Lin's CCC (Lin 1989)
nrmse(y_true, y_pred)                     → dict  NRMSE (σ-norm. and range-norm.)
y_randomization_test(X, y, ...)           → dict  Y-randomisation test (1000 permutations)
format_bf10(log10_bf10)                   → str   Human-readable BF₁₀ string
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_LOG10E = math.log10(math.e)  # ≈ 0.4343


# ─────────────────────────────────────────────────────────────────────────────
# Q1 — Does this model explain the data?
# ─────────────────────────────────────────────────────────────────────────────

def jeffreys_bf_corr(r: float, n: int) -> dict:
    """BIC-Jeffreys Bayes factor for H₁: ρ ≠ 0 vs H₀: ρ = 0.

    Uses the BIC approximation (Kass & Raftery 1995; Wagenmakers 2007):

        log(BF₁₀) = (t² − ln n) / 2,   t = r √[(n−2)/(1−r²)]

    This is numerically indistinguishable from the exact Jeffreys integral
    (Ly et al. 2015) for n ≥ 50.  Results are reported as log₁₀(BF₁₀).

    Also returns the 95 % CI on ρ via the Fisher z-transform:

        SE(z) = 1 / √(n − 3),   CI = tanh(arctanh(r) ± 1.96 × SE)

    Parameters
    ----------
    r : Pearson correlation coefficient
    n : number of observations

    Returns
    -------
    dict with keys ``log10_bf10``, ``ci95_lo``, ``ci95_hi``

    References
    ----------
    Ly, A., Verhagen, A.J., & Wagenmakers, E.-J. (2015). Journal of
    Mathematical Psychology, 29, 19–32. doi:10.1016/j.jmp.2015.06.004

    Kass, R.E. & Raftery, A.E. (1995). Bayes Factors. Journal of the
    American Statistical Association, 90(430), 773–795.
    """
    r = float(r)
    n = int(n)
    _nan = {"log10_bf10": float("nan"), "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    if n < 4 or not math.isfinite(r):
        return _nan

    r_c = max(-1.0 + 1e-9, min(1.0 - 1e-9, r))

    # BIC approximation
    t2 = r_c**2 * (n - 2) / (1.0 - r_c**2)
    log_bf10 = (t2 - math.log(n)) / 2.0
    log10_bf10 = log_bf10 * _LOG10E

    # 95 % CI on ρ via Fisher z-transform
    z = math.atanh(r_c)
    se = 1.0 / math.sqrt(max(n - 3, 1))
    return {
        "log10_bf10": log10_bf10,
        "ci95_lo": float(math.tanh(z - 1.96 * se)),
        "ci95_hi": float(math.tanh(z + 1.96 * se)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Q2 — Do two models differ in how well they explain the data?
# ─────────────────────────────────────────────────────────────────────────────

def compare_correlations_bf(
    r12: float,
    r13: float,
    r23: float,
    n: int,
) -> dict:
    """BIC Bayes factor for H₁: ρ₁₂ ≠ ρ₁₃ (two dependent correlations, same truth).

    Uses the Hotelling-Williams t-statistic for dependent correlations
    (Steiger 1980):

        t = (r₁₂ − r₁₃) √[(n−3)(1 + r₂₃) / (2 h)]
        h = 1 − r₁₂² − r₁₃² − r₂₃² + 2 r₁₂ r₁₃ r₂₃

    Converted to a BF via the BIC approximation:

        log(BF) = (t² − ln(n−3)) / 2

    Parameters
    ----------
    r12 : correlation of model-1 predictions with experimental values
    r13 : correlation of model-2 predictions with experimental values
    r23 : correlation between model-1 and model-2 predictions
    n   : number of matched observations

    Returns
    -------
    dict with keys ``log10_bf``, ``t``, ``interpretation``

    References
    ----------
    Steiger, J.H. (1980). Tests for comparing elements of a correlation
    matrix. Psychological Bulletin, 87, 245–251.
    doi:10.1037/0033-2909.87.2.245
    """
    r12, r13, r23 = float(r12), float(r13), float(r23)
    n = int(n)
    _nan = {"log10_bf": float("nan"), "t": float("nan"), "interpretation": "n/a"}
    if n < 6 or not all(math.isfinite(v) for v in (r12, r13, r23)):
        return _nan

    h = 1.0 - r12**2 - r13**2 - r23**2 + 2.0 * r12 * r13 * r23
    if h <= 0 or (1.0 + r23) <= 1e-9:
        return _nan

    denom = 2.0 * h / ((n - 3) * (1.0 + r23))
    if denom <= 0:
        return _nan

    t = (r12 - r13) / math.sqrt(denom)
    df = n - 3
    log_bf = (t**2 - math.log(max(df, 1))) / 2.0
    log10_bf = log_bf * _LOG10E

    if log10_bf > 1.5:
        interp = f"strong evidence for difference (log10BF={log10_bf:.1f})"
    elif log10_bf > 0.5:
        interp = f"moderate evidence for difference (log10BF={log10_bf:.1f})"
    elif log10_bf < -0.5:
        interp = f"evidence for equivalence (log10BF={log10_bf:.1f})"
    else:
        interp = f"ambiguous (log10BF={log10_bf:.1f})"

    return {"log10_bf": log10_bf, "t": t, "interpretation": interp}


# ─────────────────────────────────────────────────────────────────────────────
# Agreement metrics
# ─────────────────────────────────────────────────────────────────────────────

def lin_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's concordance correlation coefficient (CCC).

    CCC = 2·cov(y, ŷ) / (var(y) + var(ŷ) + (μ_y − μ_ŷ)²)

    Combines Pearson r (precision) with a penalty for systematic bias
    relative to the line of perfect concordance (y = ŷ).  CCC = 1 means
    perfect agreement; CCC = −1 means perfect disagreement.

    Reference
    ---------
    Lin, L.I.-K. (1989). A concordance correlation coefficient to evaluate
    reproducibility. Biometrics, 45, 255–268. doi:10.2307/2532051
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[finite], y_pred[finite]
    if len(y_true) < 2:
        return float("nan")
    mean_t = y_true.mean()
    mean_p = y_pred.mean()
    var_t = y_true.var(ddof=0)
    var_p = y_pred.var(ddof=0)
    cov = float(np.mean((y_true - mean_t) * (y_pred - mean_p)))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    if denom == 0.0:
        return float("nan")
    return float(2.0 * cov / denom)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Normalised RMSE in two variants.

    nrmse_sd    = RMSE / σ(y_true)           "coefficient of variation of RMSE"
    nrmse_range = RMSE / (max − min)(y_true)

    Returns
    -------
    dict with keys ``nrmse_sd`` and ``nrmse_range``
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[finite], y_pred[finite]
    if len(y_true) < 2:
        return {"nrmse_sd": float("nan"), "nrmse_range": float("nan")}
    rmse_val = math.sqrt(float(np.mean((y_true - y_pred) ** 2)))
    sd = float(y_true.std(ddof=0))
    rng = float(y_true.max() - y_true.min())
    return {
        "nrmse_sd":    (rmse_val / sd)  if sd  > 0 else float("nan"),
        "nrmse_range": (rmse_val / rng) if rng > 0 else float("nan"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Y-randomisation test
# ─────────────────────────────────────────────────────────────────────────────

def y_randomization_test(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict:
    """Y-randomisation (label-permutation) test for Ridge regression.

    Fits a Ridge model with ``n_splits``-fold CV on ``n_permutations`` random
    permutations of ``y``.  The fraction of permutation R² values that equal
    or exceed the observed R² gives the empirical p-value.

    Parameters
    ----------
    X              : feature matrix (n_samples × n_features)
    y              : target vector — real labels
    n_permutations : number of shuffles (default 1000)
    n_splits       : CV folds per permutation (default 5)
    random_state   : seed for reproducibility

    Returns
    -------
    dict with keys ``observed_r2``, ``observed_ccc``, ``perm_r2_mean``,
    ``perm_r2_std``, ``p_value``, ``perm_ccc_mean``
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(random_state)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def _make_pipe(alpha: float):
        return make_pipeline(
            StandardScaler(with_mean=False),
            Ridge(alpha=float(alpha), fit_intercept=True, solver="sag", max_iter=8000, tol=1e-4, random_state=random_state),
        )

    def _select_alpha(X_fit: np.ndarray, y_fit: np.ndarray) -> float:
        alphas = np.logspace(-3, 4, 30)
        inner_kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        best_alpha = float(alphas[0])
        best_r2 = -np.inf

        for alpha in alphas:
            fold_pred = np.zeros_like(y_fit, dtype=np.float64)
            for inner_train, inner_test in inner_kf.split(X_fit):
                model = _make_pipe(float(alpha))
                model.fit(X_fit[inner_train], y_fit[inner_train])
                fold_pred[inner_test] = model.predict(X_fit[inner_test])
            score = float(r2_score(y_fit, fold_pred))
            if score > best_r2:
                best_r2 = score
                best_alpha = float(alpha)

        return best_alpha

    def _cross_val_predict_selected_alpha(X_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
        alpha = _select_alpha(X_fit, y_fit)
        y_pred = np.zeros_like(y_fit, dtype=np.float64)
        for train_idx, test_idx in kf.split(X_fit):
            model = _make_pipe(alpha)
            model.fit(X_fit[train_idx], y_fit[train_idx])
            y_pred[test_idx] = model.predict(X_fit[test_idx])
        return y_pred

    # Observed performance
    y_cv_obs = _cross_val_predict_selected_alpha(X, y)
    obs_r2 = float(r2_score(y, y_cv_obs))
    obs_ccc = lin_ccc(y, y_cv_obs)

    # Permutation distribution
    perm_r2: list[float] = []
    perm_ccc_vals: list[float] = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        y_cv_perm = _cross_val_predict_selected_alpha(X, y_perm)
        perm_r2.append(float(r2_score(y_perm, y_cv_perm)))
        perm_ccc_vals.append(lin_ccc(y_perm, y_cv_perm))

    perm_r2_arr = np.array(perm_r2)
    p_value = float((perm_r2_arr >= obs_r2).mean())

    return {
        "observed_r2":   round(obs_r2, 4),
        "observed_ccc":  round(obs_ccc, 4) if math.isfinite(obs_ccc) else float("nan"),
        "perm_r2_mean":  round(float(perm_r2_arr.mean()), 4),
        "perm_r2_std":   round(float(perm_r2_arr.std()),  4),
        "p_value":       round(p_value, 4),
        "perm_ccc_mean": round(float(np.nanmean(perm_ccc_vals)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_bf10(log10_bf10: float) -> str:
    """Format a log₁₀(BF₁₀) value as a human-readable string.

    Examples
    --------
    >>> format_bf10(310)   ">10^300"
    >>> format_bf10(6.4)   "10^6.4"
    >>> format_bf10(-2.1)  "1/10^2.1"
    >>> format_bf10(0.05)  "≈1"
    """
    if not math.isfinite(log10_bf10):
        return "n/a"
    if log10_bf10 > 300:
        return ">10^300"
    if log10_bf10 > 0.1:
        return f"10^{log10_bf10:.1f}"
    if log10_bf10 < -0.1:
        return f"1/10^{abs(log10_bf10):.1f}"
    return "≈1"
