# LUFlow Datasets

The CSV files in the `datasets/` subdirectory are **not tracked in Git** (too large — ~1.5 GB+).  
You must download them manually before running the LUFlow experiments.

---

## Download Instructions

The LUFlow dataset is published by Lancaster University's Cyber Security & Machine Learning group:  
**Repository:** <https://github.com/LU-CSML/LUFlow>

### Required Files

Download the day-files listed below and place them in `src/data/LUFlow_Network_Intrusion/datasets/`.

#### January 2021 (`2021/01/`) — 8 files

- `2021.01.01.csv`
- `2021.01.02.csv`
- `2021.01.03.csv`
- `2021.01.04.csv`
- `2021.01.05.csv`
- `2021.01.06.csv`
- `2021.01.07.csv`
- `2021.01.08.csv`

#### February 2021 (`2021/02/`) — 17 files

- `2021.02.01.csv`
- `2021.02.02.csv`
- `2021.02.03.csv`
- `2021.02.04.csv`
- `2021.02.05.csv`
- `2021.02.06.csv`
- `2021.02.07.csv`
- `2021.02.08.csv`
- `2021.02.09.csv`
- `2021.02.10.csv`
- `2021.02.11.csv`
- `2021.02.12.csv`
- `2021.02.13.csv`
- `2021.02.14.csv`
- `2021.02.15.csv`
- `2021.02.16.csv`
- `2021.02.17.csv`

#### June 2022 (`2022/06/`) — 3 files

- `2022.06.12.csv`
- `2022.06.13.csv`
- `2022.06.14.csv`

### Quick Verification

After downloading, you should have **28** `.csv` files in the `datasets/` folder:

```powershell
Get-ChildItem src/data/LUFlow_Network_Intrusion/datasets/*.csv | Measure-Object   # should show 28 files
```

---

## Dataset Summary

| Period | Days | Total Rows | Malicious % (binary) |
|--------|------|-----------|----------------------|
| Jan 2021 | 8 | ~6.3 M | 0–52 % per day |
| Feb 2021 | 17 | ~14 M | 22–44 % per day |
| Jun 2022 | 3 | ~1.1 M | 0–58 % per day |
| **Combined** | **28** | **~21 M** | **~24 %** |

---

## How the Data Is Used

Both `luflow_fitness_check.py` and `luflow_main.py` load CSVs from `src/data/LUFlow_Network_Intrusion/datasets/`.

### Features (11 columns)

| Feature | Description |
|---|---|
| `avg_ipt` | Average inter-packet time |
| `bytes_in` | Bytes received |
| `bytes_out` | Bytes sent |
| `dest_port` | Destination port |
| `entropy` | Payload entropy |
| `num_pkts_out` | Packets sent |
| `num_pkts_in` | Packets received |
| `proto` | Protocol number |
| `src_port` | Source port |
| `total_entropy` | Total entropy |
| `duration` | Flow duration |

### Labels

Only rows with `label ∈ {benign, malicious}` are used (binary classification).

### Pool Configurations (used as "seeds" in the experiment)

The experiment constructs three **pool-pair** configurations by filtering days on their malicious-class percentage. Each pair provides pre-drift and post-drift data pools:

| Config | Pre-drift pool | Post-drift pool | Shift type |
|--------|---------------|----------------|------------|
| Seed 1 | Jan 2021 low-mal days (≤ 5 %) | Feb 2021 high-mal days (≥ 15 %) | Class-balance shift |
| Seed 2 | Jan 2021 high-mal days (≥ 15 %) | Feb 2021 high-mal days (≥ 15 %) | Feature drift (similar balance) |
| Seed 3 | Jan 2021 low-mal days (≤ 5 %) | Feb 2021 extreme-mal days (≥ 40 %) | Extreme class-balance shift |

### Stream Construction

Each run builds a **50,000-sample stream** with drift injected at t = 25,000:

- **Abrupt** — hard switch from pre-pool to post-pool at the drift point.
- **Gradual** — linear blend over a 5,000-step transition window.
- **Recurring** — concept alternates between post-pool and pre-pool every 5,000 steps after the drift point.

Features are standardized with `StandardScaler` after stream assembly.

---

## Citation

> Lancaster University Cyber Security & Machine Learning Group. *LUFlow: A Flow-Based Intrusion Detection Dataset.*  
> <https://github.com/LU-CSML/LUFlow>
