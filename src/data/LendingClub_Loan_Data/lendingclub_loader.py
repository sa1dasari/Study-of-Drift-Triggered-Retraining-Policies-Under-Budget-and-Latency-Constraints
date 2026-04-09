"""
LendingClub Loan Data -- Loader & Preprocessor.

Provides helpers to load the accepted-loans CSV, apply origination-time-only
feature selection, parse temporal columns, and produce (X, y) arrays for
specific year cohorts.

Key design decisions:
  * Only origination-time features are retained (no post-origination leakage).
  * Binary label: Charged Off = 1, Fully Paid = 0 (all other loan_status
    values are dropped).
  * ``issue_d`` is parsed with ``pd.to_datetime(df['issue_d'], format='%b-%Y')``
    and used to assign each loan to a calendar year.
  * ``emp_length`` is converted to a numeric scale (0--10, NaN -> 0).
  * ``home_ownership`` and ``purpose`` are one-hot encoded.
  * ``int_rate`` is stripped of '%' if stored as string.

Usage:
    from src.data.LendingClub_Loan_Data.lendingclub_loader import (
        load_lendingclub, get_year_cohort,
    )
    df = load_lendingclub()                       # full cleaned DataFrame
    X, y = get_year_cohort(df, year=2013)         # numpy arrays for 2013
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
#  CONSTANTS
# ----------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "datasets"
CSV_FILE = DATA_DIR / "accepted_2007_to_2018Q4.csv"

# Origination-time numeric features (known at the time the loan is issued).
NUMERIC_FEATURES = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "fico_range_low",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "inq_last_6mths",
    "delinq_2yrs",
]

# Categorical features -> will be one-hot encoded.
CATEGORICAL_FEATURES = [
    "home_ownership",
    "purpose",
]

# ``emp_length`` needs special parsing before it becomes numeric.
EMP_LENGTH_COL = "emp_length"

# Columns to read from the CSV (keeps memory manageable).
USE_COLS = (
    ["loan_status", "issue_d"]
    + NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + [EMP_LENGTH_COL]
)

# Binary label mapping.
LABEL_MAP = {"Fully Paid": 0, "Charged Off": 1}


# ----------------------------------------------------------------------
#  INTERNAL HELPERS
# ----------------------------------------------------------------------

def _parse_emp_length(series: pd.Series) -> pd.Series:
    """Convert ``emp_length`` strings ('< 1 year', '10+ years', ...) to int."""
    s = series.astype(str)
    s = s.str.replace(r"\+?\s*years?", "", regex=True)
    s = s.str.replace("< 1", "0", regex=False)
    s = s.str.strip()
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def _parse_int_rate(series: pd.Series) -> pd.Series:
    """Strip '%' and convert to float (handles both str and float input)."""
    if series.dtype == object:
        return pd.to_numeric(series.str.replace("%", "", regex=False), errors="coerce")
    return series


# ----------------------------------------------------------------------
#  PUBLIC API
# ----------------------------------------------------------------------

def load_lendingclub(max_rows: int | None = None) -> pd.DataFrame:
    """Load and preprocess the LendingClub accepted-loans CSV.

    Steps:
        1. Read only the columns we need (USE_COLS).
        2. Keep only Fully-Paid / Charged-Off rows.
        3. Parse ``issue_d`` -> datetime, extract ``issue_year``.
        4. Clean ``emp_length`` and ``int_rate``.
        5. One-hot encode categoricals.
        6. Fill remaining NaNs with 0.

    Args:
        max_rows: Optional cap on rows read (useful for quick testing).

    Returns:
        pd.DataFrame with columns:
            - ``target`` (int): 0 = Fully Paid, 1 = Charged Off
            - ``issue_year`` (int): Calendar year of issuance
            - all feature columns (numeric + one-hot encoded)
    """
    print(f"  Loading {CSV_FILE.name} ...")
    kwargs = {"usecols": USE_COLS, "low_memory": False}
    if max_rows is not None:
        kwargs["nrows"] = max_rows
    df = pd.read_csv(str(CSV_FILE), **kwargs)

    # -- Binary label filter ------------------------------------------
    df = df[df["loan_status"].isin(LABEL_MAP.keys())].copy()
    df["target"] = df["loan_status"].map(LABEL_MAP).astype(int)
    df.drop(columns=["loan_status"], inplace=True)

    # -- Temporal parsing ---------------------------------------------
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["issue_year"] = df["issue_d"].dt.year
    df.drop(columns=["issue_d"], inplace=True)

    # -- Feature cleaning ---------------------------------------------
    df["int_rate"] = _parse_int_rate(df["int_rate"])
    df[EMP_LENGTH_COL] = _parse_emp_length(df[EMP_LENGTH_COL])

    # -- One-hot encode categoricals ----------------------------------
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

    # -- Fill remaining NaN with 0 ------------------------------------
    df = df.fillna(0)

    # -- Ensure all feature columns are float -------------------------
    feature_cols = [c for c in df.columns if c not in ("target", "issue_year")]
    df[feature_cols] = df[feature_cols].astype(np.float64)

    print(f"  Loaded {len(df):,} rows  "
          f"(years {df['issue_year'].min()}--{df['issue_year'].max()})  "
          f"default rate = {100 * df['target'].mean():.1f}%")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes target & issue_year)."""
    return [c for c in df.columns if c not in ("target", "issue_year")]


def get_year_cohort(
    df: pd.DataFrame,
    year: int,
    max_samples: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) arrays for a single calendar year.

    Args:
        df:           The preprocessed DataFrame from ``load_lendingclub()``.
        year:         The issuance year to filter on.
        max_samples:  Optional cap (rows are shuffled-sampled via ``rng.choice``).
        random_state: Seed for the RNG used when sub-sampling.  Vary this
                      across experiment seeds to avoid getting the same 25 K
                      rows every time.

    Returns:
        (X, y) where X is float64 (N, D) and y is int (N,).
    """
    cohort = df[df["issue_year"] == year]
    if max_samples is not None and len(cohort) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(cohort), size=max_samples, replace=False)
        cohort = cohort.iloc[idx]
    feat_cols = get_feature_columns(df)
    X = cohort[feat_cols].values.astype(np.float64)
    y = cohort["target"].values.astype(int)
    return X, y

