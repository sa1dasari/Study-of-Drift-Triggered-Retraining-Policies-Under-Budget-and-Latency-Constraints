# LendingClub Loan Data

The CSV file in the `datasets/` subdirectory is **not tracked in Git** (too large — ~1.6 GB).  
You must download it manually before running the LendingClub experiments.

---

## Download Instructions

The LendingClub dataset is published on Kaggle:  
**Source:** <https://www.kaggle.com/datasets/wordsforthewise/lending-club>

### Required File

Download the file below and place it in `src/data/LendingClub_Loan_Data/datasets/`:

- `accepted_2007_to_2018Q4.csv`

> Only the **accepted loans** file is needed. The rejected-loans file is not used.

### Quick Verification

After downloading, you should have the CSV in place:

```powershell
Get-Item src/data/LendingClub_Loan_Data/datasets/accepted_2007_to_2018Q4.csv   # ~1.6 GB
```

---

## Dataset Summary

| Attribute | Value |
|---|---|
| **Rows (raw)** | ~2.26 M accepted loans |
| **Rows (after filtering)** | ~1.35 M (Fully Paid + Charged Off only) |
| **Columns (raw)** | ~150 |
| **Columns (used)** | 16 origination-time features -> 34 after one-hot encoding |
| **Time span** | 2007 -- 2018 (monthly via `issue_d`) |
| **Default rate** | ~20% overall (Charged Off) |

### Year-Cohort Sizes (after label filtering)

| Year | Rows | Default Rate |
|---|---|---|
| 2007 | ~25 K | 16% |
| 2008 | ~50 K | 22% |
| 2009 | ~36 K | 17% |
| 2010 | ~45 K | 14% |
| 2011 | ~72 K | 15% |
| 2012 | ~111 K | 14% |
| 2013 | ~175 K | 15% |
| 2014 | ~188 K | 18% |
| 2015 | ~199 K | 20% |
| 2016 | ~199 K | 23% |
| 2017 | ~167 K | 22% |
| 2018 | ~78 K | 21% |

---

## Drift Characteristics

The drift in this dataset is **real-world feature-space drift**, not synthetic injection. Between 2012 and 2016, LendingClub deliberately changed their underwriting policy:

- Average borrower FICO dropped from **716 to 703**
- Average interest rates rose from **10.8% to 13%**
- Correlation between FICO scores and loan grades fell from **80% (2007) to 35% (2014--2015)** as they incorporated alternative data
- Expanded into riskier borrower segments

A model trained on one year-cohort sees a genuinely different joint distribution of DTI, FICO, interest rate, and default probability when tested on a later cohort. `partial_fit` cannot absorb this because the **decision boundary itself has shifted** — not just the proportion of positives.

---

## How the Data Is Used

Both `lendingclub_fitness_check.py` and `lendingclub_main.py` load the CSV via `lendingclub_loader.py`.

### Features (16 origination-time columns)

Only features known **at the time of loan issuance** are used. All post-origination columns (`last_pymnt_d`, `total_pymnt`, `recoveries`, `collection_recovery_fee`, etc.) are excluded to prevent data leakage.

#### Numeric Features (13)

| Feature | Description |
|---|---|
| `loan_amnt` | Funded loan amount |
| `int_rate` | Interest rate (%) |
| `installment` | Monthly payment |
| `annual_inc` | Borrower annual income |
| `dti` | Debt-to-income ratio |
| `fico_range_low` | Lower bound of FICO range |
| `open_acc` | Number of open credit lines |
| `pub_rec` | Number of derogatory public records |
| `revol_bal` | Revolving balance |
| `revol_util` | Revolving utilization rate (%) |
| `total_acc` | Total number of credit lines |
| `inq_last_6mths` | Inquiries in last 6 months |
| `delinq_2yrs` | Delinquencies in last 2 years |

#### Special Numeric (1)

| Feature | Description |
|---|---|
| `emp_length` | Employment length (parsed from strings like "10+ years" to 0--10 int) |

#### Categorical Features (2, one-hot encoded)

| Feature | Description |
|---|---|
| `home_ownership` | RENT, OWN, MORTGAGE, OTHER, etc. |
| `purpose` | debt_consolidation, credit_card, home_improvement, etc. |

### Labels

- `Fully Paid` -> 0
- `Charged Off` -> 1
- All other `loan_status` values are **dropped** (Current, Late, In Grace Period, etc.)

### Temporal Ordering

The `issue_d` column (format `%b-%Y`, e.g. `Dec-2013`) is parsed to extract `issue_year`, which is used to partition loans into annual cohorts.

---

## Pool Configurations (used as "seeds" in the experiment)

The experiment uses three **year-pair** configurations. Each pair provides pre-drift and post-drift data pools drawn from different calendar years:

| Config | Pre-drift year | Post-drift year | Gap | Shift description |
|---|---|---|---|---|
| Seed 1 | 2013 | 2016 | 3 years | Maximum policy shift |
| Seed 2 | 2014 | 2016 | 2 years | Moderate drift |
| Seed 3 | 2013 | 2015 | 2 years | Different cohort pair (no overlap with Seed 2) |

Each pool is **shuffled-sampled** via `rng.choice` (25,000 rows per pool) with a per-seed `random_state` so that seeds sharing a year (e.g. seeds 1 & 3 both use 2013) draw different subsets.

### Stream Construction

Each run builds a **50,000-sample stream** with drift injected at t = 25,000:

- **Abrupt** — hard switch from pre-pool to post-pool at the drift point.
- **Gradual** — linear blend over a 5,000-step transition window.
- **Recurring** — concept alternates between post-pool and pre-pool every 5,000 steps after the drift point.

Features are standardized with `StandardScaler` after stream assembly.

---

## Citation

> LendingClub loan data (2007--2018). Originally published by LendingClub Corporation.  
> Kaggle mirror: <https://www.kaggle.com/datasets/wordsforthewise/lending-club>

