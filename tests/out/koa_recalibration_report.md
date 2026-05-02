# logKoa Recalibration Report

Least-squares refit of 7 Koa parameter entries against the S02 training set (1 983 molecules).  
The `logKow` parameters are **unchanged**. Only the Koa CSV differs.

---

## S02 performance metrics (logKoa)

| Metric   | Original | Recalibrated | Δ          |
|----------|:--------:|:------------:|:----------:|
| N        | 1 983    | 1 983        | —          |
| R²       | 0.398    | 0.650        | **+0.252** |
| RMSE     | 2.319    | 1.767        | **−0.551** |
| MAE      | 1.295    | 1.031        | **−0.265** |
| Pearson  | 0.725    | 0.833        | **+0.108** |

Scatter plots: `koa_orig_vs_recal_scatter.png`

---

## Recalibrated coefficients

Entries with the largest residual-vs-count correlation in the post-patch audit.  
All other entries (230 rows) are identical between the two files.

| Entry | Atom Type    | Neighbours   | Function           | Original | Recalibrated | Δ       | N present |
|------:|:-------------|:-------------|:------------------:|:--------:|:------------:|:-------:|:---------:|
|  20   | C sp3        | H2N2         | —                  |  4.89    |   6.33       | +1.44   |  3        |
| 141   | N sp3        | HC2          | —                  | −5.94    |  −2.19       | +3.75   | 57        |
| 142   | N sp3        | HC2(pi)      | —                  | −2.38    |  +1.04       | +3.42   | 52        |
| 155   | N sp3        | C2N(pi)      | —                  | −2.54    |  −3.38       | −0.84   | 15        |
| 156   | N sp3        | C2N(+)(pi)   | —                  | −1.93    |  +1.10       | +3.03   |  2        |
| 169   | N sp2        | N=O          | —                  | −2.02    |  −1.31       | +0.71   | 17        |
| 231   | H            | H Acceptor   | `_count_h_acceptor`| −1.51    |  −0.05       | +1.46   | 327       |

### Notes

- **Entries 141 & 142** (secondary sp3 amines) drove the largest systematic underprediction of Koa for N-containing compounds. The original coefficients were calibrated before the `_count_h_acceptor` overcounting bug was fixed; correcting that bug shifted the model, making the amine terms appear more negative than they should be.
- **Entry 231** (H-acceptor correction) converges near zero after the binary-presence patch. The patch already corrected the overcounting; the LS refit confirms the residual coefficient contribution is negligible.
- **Entry 156** (`N sp3, C2N(+)(pi)`) is present in only 2 molecules — treat with caution.
- The refit was performed with `numpy.linalg.lstsq` (rank-7 full-rank solution); sklearn `LinearRegression` crashed on this platform (suspected stack overflow in the BLAS solver for this specific matrix shape).
