# A Software-Only Thermal Proxy for Edge AI Inference (Raspberry Pi 5)

Reproducible artifact for **"CPU Utilization as a Software-Only Thermal Proxy for Multi-Tenant
Edge AI Inference: A 13-Hour Characterization and Coupling Law on Raspberry Pi 5"**
(submitted to IEEE Embedded Systems Letters).

## Summary
Thermal-aware control of edge inference usually assumes an instrumented power sensor, which most
deployed single-board computers lack. From a 13-hour measurement campaign on Raspberry Pi 5
(Arm Cortex-A76), we show that CPU utilization alone — a fully software-observable signal —
predicts SoC temperature through a compact, cross-validated coupling law:

```
T ≈ 45.6 °C + 0.175 °C/% · U     (R² = 0.93, slope 95% CI [0.1746, 0.1750], n = 232,136)
```

The slope is consistent across four independent experiment blocks, and the thermal response
settles within one control interval, so the law is directly usable for real-time thermal
control with no power instrumentation.

## Dataset
`data/` contains the de-duplicated real telemetry (Apache Parquet, 5 Hz), one collector file per
experiment block: single-workload characterization, load sweep, two-tenant concurrency, and a
6-hour stability run. Columns include SoC temperature, per-core CPU utilization, and memory.

> Scope note (stated in the paper): this deployment had **no power sensor** attached, so no board
> power is reported — the utilization law is offered as the software-only alternative. The CPU
> governor was pinned (`performance`), so the law characterizes the fixed-frequency regime.

## Contents
```
paper/main_v2_rewrite.pdf   the paper
data/*.parquet              de-duplicated 232,136-sample telemetry (4 blocks)
scripts/paper8_deep.py      coupling-law fit, leave-one-block-out CV, dynamic time constant,
                            multi-tenant characterization
scripts/paper8_figs.py      regenerates the two figures
figures/                    coupling_law.pdf, slope_consistency.pdf
LICENSE                     MIT
```

## Reproduce
```bash
pip install pandas pyarrow numpy scipy matplotlib
python scripts/paper8_deep.py   # every number in the paper
python scripts/paper8_figs.py   # the figures
```

## Citation
Manu Nicholas Jacob, IEEE Embedded Systems Letters, 2026. MIT License.
