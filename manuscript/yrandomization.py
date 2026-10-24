"""yrandomization.py
===================
Y-randomisation of the two ridge models (Crippen and the combined
Crippen+Naef & Acree model) for both endpoints, as described in SI section
S5.2: the ridge penalty is selected once on the real data with the same inner
5-fold CV as the benchmark, then the target values are permuted 300 times and
out-of-fold predictions are obtained on the same outer 5-fold split, with the
penalty fixed and the lsqr solver. The observed R2 uses the same procedure on
the unpermuted targets.

Writes cache/yrandomization_results.csv and
cache/yrandomization_null_<model>_<endpoint>.csv (read by generate_figures.py
for SI Figure S1).

Usage:
    python yrandomization.py [--n-permutations 300]
"""

import argparse
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import paths
from scripts.benchmark_5model import KOA_BASE, KOW_BASE, _build_common_set

MODELS = {"crippen": "X_crippen", "naefacree_mixed": "X_mixed"}


def _ridge(alpha, solver):
    return make_pipeline(StandardScaler(with_mean=False),
                         Ridge(alpha=float(alpha), fit_intercept=True, solver=solver, max_iter=8000,
                               tol=1e-4, random_state=42))


def select_alpha(X, y):
    """Inner 5-fold CV over 30 log-spaced alphas, as in the benchmark ridge models."""
    alphas = np.logspace(-3, 4, 30)
    splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))
    rmses = []
    for alpha in alphas:
        pred = np.zeros_like(y, dtype=np.float64)
        for tr, te in splits:
            pred[te] = _ridge(alpha, "lsqr").fit(X[tr], y[tr]).predict(X[te])
        rmses.append(math.sqrt(mean_squared_error(y, pred)))
    return alphas[int(np.argmin(rmses))]


def cv_r2(X, y, alpha, splits):
    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in splits:
        pred[te] = _ridge(alpha, "lsqr").fit(X[tr], y[tr]).predict(X[te])
    return r2_score(y, pred)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-permutations", type=int, default=300)
    args = ap.parse_args()
    paths.CACHE.mkdir(exist_ok=True)

    data = {
        "logKow": _build_common_set(paths.S01_SDF, "logP", "logKow", KOW_BASE, KOW_BASE),
        "logKoa": _build_common_set(paths.S02_SDF, "logKoa", "logKoa", KOA_BASE, KOA_BASE),
    }
    rows = []
    for endpoint, d in data.items():
        y = d["y"]
        splits = list(KFold(n_splits=5, shuffle=True, random_state=42).split(y))
        for mkey, xkey in MODELS.items():
            X = d[xkey]
            alpha = select_alpha(X, y)
            observed = cv_r2(X, y, alpha, splits)
            rng = np.random.default_rng(42)
            null = np.array([cv_r2(X, rng.permutation(y), alpha, splits) for _ in range(args.n_permutations)])
            pd.DataFrame({"perm_r2": null}).to_csv(paths.CACHE / f"yrandomization_null_{mkey}_{endpoint}.csv",
                                                   index=False)
            rows.append({"model": mkey, "endpoint": endpoint, "n": len(y), "alpha": alpha,
                         "observed_r2": observed, "n_permutations": args.n_permutations,
                         "perm_r2_mean": null.mean(), "perm_r2_std": null.std(),
                         "p_value": float((null >= observed).mean())})
            print(f"{endpoint} {mkey}: alpha={alpha:.3g} observed R2={observed:.4f} "
                  f"null {null.mean():.3f}+/-{null.std():.3f} p={rows[-1]['p_value']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(paths.CACHE / "yrandomization_results.csv", index=False)
    print(f"Saved {paths.CACHE / 'yrandomization_results.csv'}")


if __name__ == "__main__":
    main()
