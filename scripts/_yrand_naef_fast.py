"""Fast y-randomization test for naef using pure numpy linear regression."""
import pathlib
import csv
import numpy as np

OUT = pathlib.Path(r"c:\Users\luc\git\kawow\tests\out")
RNG = np.random.default_rng(0)
N_PERM = 1000


def _ccc(y, yp):
    m1, m2 = np.mean(y), np.mean(yp)
    s1, s2 = np.var(y), np.var(yp)
    cov = np.mean((y - m1) * (yp - m2))
    denom = s1 + s2 + (m1 - m2) ** 2
    return 2 * cov / denom if denom > 0 else 0.0


def _cv_r2(x, y, folds=5):
    n = len(y)
    idx = np.arange(n)
    fold_size = n // folds
    yp = np.zeros_like(y)
    for k in range(folds):
        if k < folds - 1:
            test_idx = idx[k * fold_size:(k + 1) * fold_size]
            train_idx = np.concatenate([idx[:k * fold_size], idx[(k + 1) * fold_size:]])
        else:
            test_idx = idx[k * fold_size:]
            train_idx = idx[:k * fold_size]
        X_tr = np.column_stack([x[train_idx], np.ones(len(train_idx))])
        b, *_ = np.linalg.lstsq(X_tr, y[train_idx], rcond=None)
        X_te = np.column_stack([x[test_idx], np.ones(len(test_idx))])
        yp[test_idx] = X_te @ b
    ss_res = np.sum((y - yp) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot, _ccc(y, yp)


TASKS = [
    ("s01_smarts_vs_experimental.csv", "logKow_exp_s01", "logKow_smarts", "logKow"),
    ("s02_smarts_vs_experimental.csv", "logKoa_exp_s02", "logKoa_smarts", "logKoa"),
]

for csv_f, ec, pc, ep in TASKS:
    rows = list(csv.DictReader(open(OUT / csv_f, encoding="utf-8")))
    pairs = []
    for r in rows:
        ev, pv = r.get(ec, ""), r.get(pc, "")
        if ev and pv:
            try:
                e, p = float(ev), float(pv)
                if np.isfinite(e) and np.isfinite(p):
                    pairs.append((e, p))
            except ValueError:
                pass
    y = np.array([e for e, _ in pairs])
    x = np.array([p for _, p in pairs])
    print(f"Processing {ep}: n={len(y)}")
    obs_r2, obs_ccc = _cv_r2(x, y)
    perm_r2s = []
    for i in range(N_PERM):
        y_perm = RNG.permutation(y)
        pr2, _ = _cv_r2(x, y_perm)
        perm_r2s.append(pr2)
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{N_PERM} done")
    perm_arr = np.array(perm_r2s)
    p_val = round(float(np.mean(perm_arr >= obs_r2)), 1)
    print(
        f"  RESULT {ep}: obs_r2={obs_r2:.4f} obs_ccc={obs_ccc:.4f} "
        f"perm_r2_mean={np.mean(perm_arr):.4f} perm_r2_std={np.std(perm_arr):.4f} "
        f"p_value={p_val}"
    )

