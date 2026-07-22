#!/usr/bin/env python3
"""Deeper analysis of Paper 8's real telemetry for a stronger honest rewrite.
Static coupling law (per-block + CV), dynamic thermal time constant, multi-tenant
characterization. Power/voltage/freq are stubs -> excluded."""
import pandas as pd, numpy as np, glob, os
from scipy import stats

RAW = "data"
# de-duplicate: keep one collector file per physical run (drop the ~90s-later twin)
files = sorted(glob.glob(os.path.join(RAW,"*.parquet")))
# group by block prefix, keep first of each near-identical pair
keep=[]
seen_sizes={}
for f in files:
    n=len(pd.read_parquet(f, columns=["temperature_c"]))
    base=os.path.basename(f).rsplit("_",2)[0]
    key=(base, round(n,-2))
    if key not in seen_sizes:
        seen_sizes[key]=f; keep.append(f)
print(f"de-duplicated {len(files)} -> {len(keep)} files (one collector per run)\n")

frames=[pd.read_parquet(f) for f in keep]
df=pd.concat(frames, ignore_index=True)
d=df[["temperature_c","cpu_percent"]].dropna()
print(f"unique samples after de-dup: {len(df):,}  (raw double-counts to 471,827)")

print("\n=== STATIC COUPLING LAW  T = a + b*U  (global) ===")
sl,ic,r,p,se=stats.linregress(d.cpu_percent, d.temperature_c)
n=len(d)
# 95% CI on slope
tcrit=stats.t.ppf(0.975, n-2)
print(f"  T = {ic:.3f} + {sl:.5f}*U   r={r:.4f} R2={r**2:.4f} n={n:,}")
print(f"  slope 95% CI: [{sl-tcrit*se:.5f}, {sl+tcrit*se:.5f}]")
resid=d.temperature_c-(ic+sl*d.cpu_percent)
print(f"  residual std: {resid.std():.3f} C")

print("\n=== CROSS-VALIDATION across blocks ===")
if "block" in df.columns:
    blocks=[b for b in df.block.dropna().unique()]
    for tb in blocks:
        tr=df[df.block!=tb][["temperature_c","cpu_percent"]].dropna()
        te=df[df.block==tb][["temperature_c","cpu_percent"]].dropna()
        if len(tr)<100 or len(te)<100: continue
        s2,i2,_,_,_=stats.linregress(tr.cpu_percent,tr.temperature_c)
        pred=i2+s2*te.cpu_percent
        rmse=np.sqrt(((te.temperature_c-pred)**2).mean())
        print(f"  train!={tb:8s} test={tb:8s}: RMSE={rmse:.3f}C  (fit T={i2:.2f}+{s2:.4f}U)")

print("\n=== DYNAMIC THERMAL MODEL: first-order  dT/dt = (Tss(U)-T)/tau ===")
# use the longest continuous run (stability), estimate tau from step response of T to U
best=max(keep, key=lambda f: len(pd.read_parquet(f, columns=["temperature_c"])))
g=pd.read_parquet(best)
g=g.dropna(subset=["temperature_c","cpu_percent","log_timestamp"]).sort_values("log_timestamp")
print(f"  using {os.path.basename(best)}  n={len(g)}")
# discrete-time fit: T[t+1]-T[t] = k*(a + b*U[t] - T[t]); k = dt/tau
T=g.temperature_c.values; U=g.cpu_percent.values; ts=g.log_timestamp.values
dt=np.median(np.diff(ts))
dT=np.diff(T)
Tss = ic+sl*U[:-1]                      # steady-state target from static law
xreg=(Tss - T[:-1])
mask=np.abs(xreg)>0.05
if mask.sum()>100:
    k,kc,kr,kp,kse=stats.linregress(xreg[mask], dT[mask])
    tau=dt/k if k>0 else float('nan')
    print(f"  dt={dt:.2f}s  k={k:.4f}  ->  tau ~= {tau:.1f} s  (thermal time constant)")
    print(f"  interpretation: temperature reaches ~63% of a step change in ~{tau:.0f}s")

print("\n=== MULTI-TENANT (concurrent block) ===")
if "workload2_model" in df.columns:
    conc=df[df.workload2_model.notna()]
    single=df[df.workload2_model.isna() & df.workload1_model.notna()] if "workload1_model" in df.columns else pd.DataFrame()
    if len(conc):
        print(f"  concurrent samples: {len(conc):,}")
        print(f"  concurrent  temp mean={conc.temperature_c.mean():.1f}C  cpu={conc.cpu_percent.mean():.1f}%  mem={conc.memory_used_mb.mean():.0f}MB")
        if len(single):
            print(f"  single      temp mean={single.temperature_c.mean():.1f}C  cpu={single.cpu_percent.mean():.1f}%  mem={single.memory_used_mb.mean():.0f}MB")
        print(f"  concurrent workload pairs: {conc.groupby(['workload1_model','workload2_model']).size().to_dict()}")

print("\n=== THERMAL HEADROOM ===")
print(f"  max temp {df.temperature_c.max():.1f}C vs 75C cap; violations>{75}C: {(df.temperature_c>75).sum()}")
print(f"  temp range: {df.temperature_c.min():.1f}-{df.temperature_c.max():.1f}C")
print(f"  is cpu_freq a stub? uniq={df.cpu_freq_mhz.nunique()} (constant=DVFS off)")
