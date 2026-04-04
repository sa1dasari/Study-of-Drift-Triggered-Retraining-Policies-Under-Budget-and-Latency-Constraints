# LUFlow Datasets

The CSV files in this directory are **not tracked in Git** (too large — 1.5 GB total).  
You must download them manually before running experiments.

## Download Instructions

The LUFlow dataset is published by Lancaster University's Cyber Security & Machine Learning group:  
**Repository:** <https://github.com/LU-CSML/LUFlow>

### Required Files

Download the following 21 day-files and place them in this directory (`src/data/LUFlow_datasets/`):

#### January 2021 (`2021/01/`)
- `2021.01.01.csv`
- `2021.01.02.csv`
- `2021.01.03.csv`
- `2021.01.04.csv`
- `2021.01.05.csv`
- `2021.01.06.csv`
- `2021.01.07.csv`
- `2021.01.08.csv`

#### February 2021 (`2021/02/`)
- `2021.02.01.csv`
- `2021.02.02.csv`
- `2021.02.03.csv`
- `2021.02.04.csv`
- `2021.02.05.csv`
- `2021.02.06.csv`
- `2021.02.07.csv`
- `2021.02.08.csv`
- `2021.02.09.csv`
- `2021.02.17.csv`

#### June 2022 (`2022/06/`)
- `2022.06.12.csv`
- `2022.06.13.csv`
- `2022.06.14.csv`

### Quick Verification

After downloading, you should have 21 `.csv` files totalling approximately 1.5 GB:

```
ls src/data/LUFlow_datasets/*.csv | Measure-Object   # should show 21 files
```

## Dataset Summary

| Period | Days | Total Rows | Malicious % (binary) |
|--------|------|-----------|----------------------|
| Jan 2021 | 8 | ~6.3 M | 0–52% per day |
| Feb 2021 | 10 | ~8.7 M | 22–44% per day |
| Jun 2022 | 3 | ~1.1 M | 0–58% per day |
| **Combined** | **21** | **~16.1 M** | **24.5%** |

## Citation

> Lancaster University Cyber Security & Machine Learning Group. *LUFlow: A Flow-Based Intrusion Detection Dataset.*  
> https://github.com/LU-CSML/LUFlow

