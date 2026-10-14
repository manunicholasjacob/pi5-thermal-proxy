# Telemetry data dictionary

Four Apache Parquet files, one per experiment block, sampled at 5 Hz (median interval 0.200 s)
on a Raspberry Pi 5 (4x Cortex-A76, 2 GB) with the `performance` governor (cpu_freq_mhz is
2400 in every row). 232,136 rows in total, about 13.0 hours.

| file | block | rows | span | block-specific columns |
|---|---|---:|---|---|
| matrix_block1_20260307_212741.parquet | single-workload characterization | 7,573 | 0 h 25.5 min | model, resolution, repeat |
| matrix_block2_20260307_215631.parquet | load sweep | 41,286 | 2 h 19 min | model, resolution, load_level |
| concurrent_20260308_001534.parquet | two-tenant concurrency | 73,509 | 4 h 08 min | workload1_model, workload2_model |
| stability_20260308_042347.parquet | stability run | 109,768 | 6 h 08 min | test_name, model, controller |

Columns common to every file:

| column | type | meaning |
|---|---|---|
| timestamp | string (ISO 8601) | collector start time for the file (constant within a file) |
| datetime | string (ISO 8601) | sample wall-clock time |
| log_timestamp | float | sample time as Unix seconds |
| experiment | string | experiment id (matches the file name stem) |
| block | string | block label used by the leave-one-block-out analysis: block1, block2, block3, block5 (there is no block4); empty in the first row of each file |
| temperature_c | float | SoC temperature, degrees C |
| thermal_state | string | collector's coarse label (`normal` or `warm`) |
| cpu_percent | float | whole-CPU utilization, 0 to 100 (one value for all four cores, not per core) |
| memory_percent | float | system memory in use, percent |
| memory_used_mb | float | system memory in use, MB |
| process_memory_mb | float | resident memory of the inference process, MB |
| cpu_freq_mhz | float | reported CPU clock; constant 2400 because DVFS was pinned |

No power, voltage or current columns: no power sensor was attached to this deployment, and the
placeholder columns of earlier versions were removed (see the main README, "Data revision").

Load in Python:

```python
import glob, pandas as pd
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/*.parquet"))], ignore_index=True)
```

Licence: the repository is MIT licensed, which covers these files.
