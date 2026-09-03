# A Software-Only Thermal Proxy for Edge AI Inference (Raspberry Pi 5)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21844859.svg)](https://doi.org/10.5281/zenodo.21844859)

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

> **Data revision (Sep 2026).** Earlier versions of these Parquet files carried
> `voltage_v`/`current_a`/`power_w` columns written by the collector as placeholders: no power
> sensor was attached, and forensics show the values were a random stub (constant 5.0 V, zero
> correlation with utilization or temperature, zero lag-1 autocorrelation). They contained no
> measurement and have been removed (`scripts/strip_stub_power_columns.py`) so the dataset cannot
> be mistaken for a power measurement. No analysis in the paper ever used them.
>
> Scope note (stated in the paper): this deployment read **no power sensor**, so no board
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

## State-of-the-art comparison (revision)

`scripts/paper8_sota_comparison.py` evaluates the standard temperature-prediction structures on
this dataset under the same leave-one-block-out protocol as the law: a HotSpot-class lumped RC
model driven by utilization, AR(1)/ARX forecasters on the temperature sensor, a multi-signal
static regression, and an EWMA-filtered law. Result (mean held-out RMSE): every sensor-free,
actuatable structure lands in the same 1.6-1.8 degC band as the two-constant law, while
sensor-feedback forecasters are ~2.7x tighter at one 200 ms step but cannot predict the effect of
an allocation change. Summary table: `data/sota_comparison_summary.csv`.

## Reproduce
```bash
pip install pandas pyarrow numpy scipy matplotlib
python scripts/paper8_deep.py   # every number in the paper
python scripts/paper8_figs.py   # the figures
```

## Citation
Manu Nicholas Jacob, IEEE Embedded Systems Letters, 2026. MIT License.

## Archived version

This artifact is archived on Zenodo. The concept DOI
[10.5281/zenodo.21844859](https://doi.org/10.5281/zenodo.21844859)
always resolves to the latest release, and `CITATION.cff` carries the full metadata,
which is what GitHub's "Cite this repository" button renders.
