#!/usr/bin/env python3
"""Recompute the README headline numbers from data/*.parquet and fail on drift.

The values checked are the ones the README states: the coupling law
T = 45.6 + 0.175 * U, R^2 = 0.93, slope 95% CI [0.1746, 0.1750], n = 232,136,
and a 13-hour campaign over four blocks. Run from the repository root.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

files = sorted(glob.glob(os.path.join(DATA, "*.parquet")))
frames = [pd.read_parquet(f) for f in files]
df = pd.concat(frames, ignore_index=True)
d = df[["temperature_c", "cpu_percent"]].dropna()

fit = stats.linregress(d.cpu_percent, d.temperature_c)
n = len(d)
t = stats.t.ppf(0.975, n - 2)
lo, hi = fit.slope - t * fit.stderr, fit.slope + t * fit.stderr

hours = 0.0
for f in frames:
    ts = pd.to_datetime(f["datetime"])
    hours += (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600.0

checks = [
    ("blocks", len(files), 4),
    ("n", n, 232136),
    ("intercept (1 dp)", round(fit.intercept, 1), 45.6),
    ("slope (3 dp)", round(fit.slope, 3), 0.175),
    ("R^2 (2 dp)", round(fit.rvalue ** 2, 2), 0.93),
    ("slope CI low (4 dp)", round(lo, 4), 0.1746),
    ("slope CI high (4 dp)", round(hi, 4), 0.1750),
    ("campaign hours (0 dp)", round(hours), 13),
]

failed = 0
for name, got, want in checks:
    ok = np.isclose(got, want)
    failed += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got}, README says {want}")
sys.exit(1 if failed else 0)
