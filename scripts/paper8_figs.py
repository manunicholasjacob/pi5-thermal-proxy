#!/usr/bin/env python3
"""Figures for Paper 8 rewrite: (1) coupling law hexbin+fit, (2) per-block slopes."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np, glob, os
import matplotlib
# Type-3 fonts are matplotlib's default and an explicit desk-reject trigger at DATE.
# 42 selects TrueType, which is what the venues want.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from scipy import stats

RAW = "data"
OUT = "figures"
os.makedirs(OUT, exist_ok=True)
files=sorted(glob.glob(os.path.join(RAW,"*.parquet")))
keep=[]; seen={}
for f in files:
    n=len(pd.read_parquet(f,columns=["temperature_c"])); base=os.path.basename(f).rsplit("_",2)[0]
    k=(base,round(n,-2))
    if k not in seen: seen[k]=f; keep.append(f)
df=pd.concat([pd.read_parquet(f) for f in keep], ignore_index=True)
dfit=df[["temperature_c","cpu_percent"]].dropna()
d=df[["temperature_c","cpu_percent","block"]].dropna()

plt.rcParams.update({"font.size":9,"font.family":"serif","figure.dpi":300})
ACC="#2b6cb0"; INK="#1a1a1a"

# Fig 1: coupling law
fig,ax=plt.subplots(figsize=(3.4,2.7))
hb=ax.hexbin(dfit.cpu_percent,dfit.temperature_c,gridsize=45,cmap="Blues",bins="log",mincnt=1)
sl,ic,r,p,se=stats.linregress(dfit.cpu_percent,dfit.temperature_c)
xs=np.array([0,100]); ax.plot(xs,ic+sl*xs,color="#c05621",lw=1.8,
    label=f"T = {ic:.1f} + {sl:.3f}·U\n$R^2$={r**2:.3f}, n={len(dfit):,}")
ax.set_xlabel("CPU utilization U (%)"); ax.set_ylabel("SoC temperature T (°C)")
ax.legend(fontsize=7,loc="upper left",framealpha=0.9)
ax.set_title("CPU–thermal coupling law",fontsize=9)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"coupling_law.pdf"),bbox_inches="tight")
fig.savefig(os.path.join(OUT,"coupling_law.png"),bbox_inches="tight",dpi=150)

# Fig 2: per-block slope consistency
fig2,ax2=plt.subplots(figsize=(3.4,2.7))
blocks=sorted([b for b in d.block.dropna().unique()])
slopes=[]; labels=[]
for b in blocks:
    sub=d[d.block==b]
    if len(sub)<500: continue
    s,i,rr,pp,ss=stats.linregress(sub.cpu_percent,sub.temperature_c)
    tcrit=stats.t.ppf(0.975,len(sub)-2)
    slopes.append((s,tcrit*ss)); labels.append(f"{b}\n(n={len(sub)//1000}k)")
ys=np.arange(len(slopes))
ax2.errorbar([s[0] for s in slopes],ys,xerr=[s[1] for s in slopes],fmt="o",
    color=ACC,capsize=3,ms=5)
ax2.axvline(sl,color="#c05621",ls="--",lw=1.2,label=f"global {sl:.3f}")
ax2.set_yticks(ys); ax2.set_yticklabels(labels,fontsize=7)
ax2.set_xlabel("coupling slope b (°C/%)")
ax2.set_title("Slope consistency across blocks",fontsize=9)
ax2.legend(fontsize=7); ax2.spines[["top","right"]].set_visible(False)
fig2.tight_layout(); fig2.savefig(os.path.join(OUT,"slope_consistency.pdf"),bbox_inches="tight")
fig2.savefig(os.path.join(OUT,"slope_consistency.png"),bbox_inches="tight",dpi=150)
print("wrote coupling_law.pdf, slope_consistency.pdf to",OUT)
