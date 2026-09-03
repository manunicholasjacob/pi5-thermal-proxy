#!/usr/bin/env python3
"""State-of-the-art comparison for the CPU-thermal coupling law (ESL revision).

Implements the standard alternatives for temperature prediction on the released
13-hour dataset and evaluates all of them under the same leave-one-block-out
protocol as the coupling law:

  CL        T = b0 + b1*U                      (ours: static utilization law)
  CL-EWMA   T = b0 + b1*ewma_tau(U)            (thermally filtered utilization)
  MLR       T = b0 + sum_c b_c*U_c + b_m*mem   (Isci/Bellosa-class multi-signal
                                                static model, per-core counters)
  RC        T(t+1) = T(t) + dt/tau*(Tss(U)-T)  (HotSpot-class lumped first-order
                                                RC driven by utilization)
  AR(1)     T(t+1) = c + phi*T(t)              (Coskun-class autoregressive
                                                forecaster, temp sensor only)
  ARX       T(t+1) = c + phi*T(t) + beta*U(t)  (AR + exogenous utilization)

Two evaluation targets:
  1-step prediction (200 ms horizon), the forecasting task AR-family methods
  are designed for; and steady-state/what-if prediction from software signals
  alone, the task a sensorless allocation controller actually needs, for which
  feedback forecasters are structurally unusable (they cannot answer "what
  happens if I change the allocation" and they require the very sensor whose
  absence motivates the proxy).

Run:  python scripts/paper8_sota_comparison.py
"""
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def load_blocks():
    blocks = {}
    for p in sorted(glob.glob(os.path.join(DATA, "*.parquet"))):
        name = os.path.basename(p).split("_20")[0]
        df = pd.read_parquet(p)
        blocks[name] = df
    return blocks


def cols(df):
    tcol = [c for c in df.columns if "temp" in c.lower()][0]
    ucol = [c for c in df.columns if c.lower() in ("cpu_percent", "cpu_util",
                                                   "cpu", "util", "cpu_pct")]
    if not ucol:
        ucol = [c for c in df.columns if "cpu" in c.lower() and "per" not in c.lower()]
    return tcol, ucol[0]


MLR_FEATS = ["cpu_percent", "memory_percent", "memory_used_mb"]


def ewma(x, alpha):
    y = np.empty_like(x, dtype=float)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * v + (1 - alpha) * acc
        y[i] = acc
    return y


def fit_ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def predict_ols(beta, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    return X1 @ beta


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    blocks = load_blocks()
    names = list(blocks)
    print("blocks:", {k: len(v) for k, v in blocks.items()})

    rows = []
    for held in names:
        train = pd.concat([blocks[b] for b in names if b != held], ignore_index=True)
        test = blocks[held]
        tcol, ucol = cols(test)
        Ttr, Utr = train[tcol].to_numpy(float), train[ucol].to_numpy(float)
        Tte, Ute = test[tcol].to_numpy(float), test[ucol].to_numpy(float)

        res = {"held": held}

        # ---- CL: static utilization law (ours)
        b = fit_ols(Utr.reshape(-1, 1), Ttr)
        res["CL"] = rmse(predict_ols(b, Ute.reshape(-1, 1)), Tte)

        # ---- CL-EWMA: filtered utilization, alpha fit on train grid
        best = None
        for alpha in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            bb = fit_ols(ewma(Utr, alpha).reshape(-1, 1), Ttr)
            e = rmse(predict_ols(bb, ewma(Utr, alpha).reshape(-1, 1)), Ttr)
            if best is None or e < best[0]:
                best = (e, alpha, bb)
        _, alpha, bb = best
        res["CL-EWMA"] = rmse(predict_ols(bb, ewma(Ute, alpha).reshape(-1, 1)), Tte)
        res["ewma_alpha"] = alpha

        # ---- MLR: utilization + memory signals (multi-signal static model)
        feats = [c for c in MLR_FEATS if c in train.columns and c in test.columns]
        Xtr = train[feats].to_numpy(float)
        Xte = test[feats].to_numpy(float)
        bmlr = fit_ols(Xtr, Ttr)
        res["MLR"] = rmse(predict_ols(bmlr, Xte), Tte)
        res["mlr_feats"] = len(feats)

        # ---- RC: first-order lumped model, tau grid + OLS on steady-state law
        # T(t+1) = T + dt/tau * (b0 + b1*U - T); fit b0,b1 by OLS of T on U
        # (steady state), tau by 1-step grid search on train
        dt = 0.2
        b0, b1 = b[0], b[1]
        best_tau, best_e = None, None
        for tau in (0.5, 1, 2, 5, 10, 20, 40, 80, 160):
            pred = Ttr[:-1] + dt / tau * (b0 + b1 * Utr[:-1] - Ttr[:-1])
            e = rmse(pred, Ttr[1:])
            if best_e is None or e < best_e:
                best_tau, best_e = tau, e
        pred = Tte[:-1] + dt / best_tau * (b0 + b1 * Ute[:-1] - Tte[:-1])
        res["RC_1step"] = rmse(pred, Tte[1:])
        res["rc_tau_s"] = best_tau
        # RC rollout without sensor feedback = what a sensorless controller has
        Tsim = np.empty(len(Ute))
        Tsim[0] = b0 + b1 * Ute[0]
        for i in range(1, len(Ute)):
            Tsim[i] = Tsim[i - 1] + dt / best_tau * (b0 + b1 * Ute[i - 1] - Tsim[i - 1])
        res["RC_rollout"] = rmse(Tsim, Tte)

        # ---- AR(1): temperature-only forecaster
        bar = fit_ols(Ttr[:-1].reshape(-1, 1), Ttr[1:])
        res["AR1_1step"] = rmse(predict_ols(bar, Tte[:-1].reshape(-1, 1)), Tte[1:])

        # ---- ARX: temperature + utilization
        barx = fit_ols(np.column_stack([Ttr[:-1], Utr[:-1]]), Ttr[1:])
        res["ARX_1step"] = rmse(
            predict_ols(barx, np.column_stack([Tte[:-1], Ute[:-1]])), Tte[1:])
        # ARX rollout without the sensor (feedback replaced by own prediction)
        Tsim = np.empty(len(Ute))
        Tsim[0] = b0 + b1 * Ute[0]
        for i in range(1, len(Ute)):
            Tsim[i] = barx[0] + barx[1] * Tsim[i - 1] + barx[2] * Ute[i - 1]
        res["ARX_rollout"] = rmse(Tsim, Tte)

        # CL 1-step for the same forecasting comparison
        res["CL_as_1step"] = rmse(predict_ols(b, Ute[:-1].reshape(-1, 1)), Tte[1:])

        rows.append(res)
        print(res)

    df = pd.DataFrame(rows).set_index("held")
    num = df.select_dtypes(float)
    print("\n=== leave-one-block-out RMSE (degC), mean over held-out blocks ===")
    print(num.mean().round(3).to_string())
    out = os.path.join(HERE, "..", "data", "sota_comparison_summary.csv")
    df.to_csv(out)
    print("saved:", out)


if __name__ == "__main__":
    main()
