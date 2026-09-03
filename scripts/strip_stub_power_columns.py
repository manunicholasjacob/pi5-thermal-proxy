#!/usr/bin/env python3
"""Remove the collector's placeholder power fields from the released dataset.

The telemetry collector wrote voltage_v/current_a/power_w fields even though no
power sensor was attached: voltage is a constant 5.0 and current is a random
stub (zero correlation with utilization or temperature, zero lag-1
autocorrelation). No analysis in the paper ever used them. They are removed
here so the released data cannot be mistaken for a power measurement.
"""
import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUB = ["voltage_v", "current_a", "power_w"]

for p in sorted(glob.glob(os.path.join(HERE, "..", "data", "*.parquet"))):
    df = pd.read_parquet(p)
    found = [c for c in STUB if c in df.columns]
    if not found:
        print(os.path.basename(p), "already clean")
        continue
    df = df.drop(columns=found)
    df.to_parquet(p, index=False)
    print(os.path.basename(p), "removed", found, "->", len(df.columns), "cols")
