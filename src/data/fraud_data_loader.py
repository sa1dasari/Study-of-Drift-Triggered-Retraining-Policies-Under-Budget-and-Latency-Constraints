"""
CIS Fraud Detection data loader and preprocessor.

Loads train_transaction.csv + train_identity.csv, joins on TransactionID,
sorts by TransactionDT, encodes categorical, imputes missing values,
and returns (X, y) numpy arrays ready for streaming.

The full preprocessed dataset is cached in memory so that multiple
pool slices (needed for gradual / recurring drift) are cheap.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Columns to drop entirely (IDs, target, time axis, high-cardinality text)
_DROP_COLS = [
    "TransactionID", "TransactionDT", "isFraud",
    "P_emaildomain", "R_emaildomain", "DeviceInfo",
]

# Transaction categorical columns → label-encoded
_TXN_CAT_COLS = [
    "ProductCD", "card4", "card6",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

# Identity categorical columns → label-encoded
_ID_CAT_COLS = [
    "DeviceType",
    "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18",
    "id_19", "id_20", "id_21", "id_22", "id_23", "id_24", "id_25",
    "id_26", "id_27", "id_28", "id_29", "id_30", "id_31", "id_32",
    "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
]

_ALL_CAT_COLS = _TXN_CAT_COLS + _ID_CAT_COLS


class FraudDataLoader:
    """
    Loads and preprocesses the CIS Fraud Detection dataset once,
    then serves arbitrary time-ordered slices via get_pool().

    Usage:
        loader = FraudDataLoader("src/data/CIS Fraud Detection")
        X_pre, y_pre = loader.get_pool(start_offset=0, n_samples=50000)
        X_post, y_post = loader.get_pool(start_offset=50000, n_samples=50000)
    """

    def __init__(self, data_dir):
        """
        Args:
            data_dir (str | Path): Directory containing the 4 CSV files.
        """
        self.data_dir = Path(data_dir)
        self._X_full = None   # cached after first load
        self._y_full = None
        self._n_total = 0

    def get_pool(self, start_offset=0, n_samples=50_000):
        """
        Return a contiguous time-ordered slice of the preprocessed data.

        Args:
            start_offset (int): First row index (0-based, after sorting).
            n_samples (int): Number of rows in the slice.

        Returns:
            (X, y): numpy arrays — X is float64 (n_samples, n_features),
                    y is int {0, 1} (n_samples,).

        Raises:
            ValueError: If the requested window exceeds available rows.
        """
        self._ensure_loaded()

        end = start_offset + n_samples
        if end > self._n_total:
            raise ValueError(
                f"Requested rows [{start_offset}, {end}) but dataset has "
                f"only {self._n_total} rows. Reduce n_samples or start_offset."
            )

        return (
            self._X_full[start_offset:end].copy(),
            self._y_full[start_offset:end].copy(),
        )

    @property
    def total_rows(self):
        """Total number of rows available after preprocessing."""
        self._ensure_loaded()
        return self._n_total

    def _ensure_loaded(self):
        """Load and preprocess once, then cache."""
        if self._X_full is not None:
            return
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """
        Full preprocessing pipeline:
        1. Load CSVs
        2. Left-join on TransactionID
        3. Sort by TransactionDT
        4. Label-encode categorical
        5. Drop unwanted columns
        6. Impute missing values
        7. Convert to numpy
        """
        print("Loading CIS Fraud Detection data (this happens once)...")

        # 1. Load CSVs
        txn_path = self.data_dir / "train_transaction.csv"
        id_path = self.data_dir / "train_identity.csv"

        df_txn = pd.read_csv(txn_path)
        df_id = pd.read_csv(id_path)

        print(f"  Transactions : {len(df_txn):,} rows, {len(df_txn.columns)} cols")
        print(f"  Identity     : {len(df_id):,} rows, {len(df_id.columns)} cols")

        # 2. Join on transaction IDs
        df = df_txn.merge(df_id, on="TransactionID", how="left")
        print(f"  After join   : {len(df):,} rows, {len(df.columns)} cols")

        # 3. Sort by time axis
        df.sort_values("TransactionDT", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 4. Extract target variable before dropping it
        y = df["isFraud"].values.astype(int)

        # 5. Label-encode categorical
        for col in _ALL_CAT_COLS:
            if col not in df.columns:
                continue
            # Fill NaN with a sentinel string before encoding
            df[col] = df[col].fillna("__MISSING__")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        # 6. Drop unwanted columns
        drop = [c for c in _DROP_COLS if c in df.columns]
        df.drop(columns=drop, inplace=True)

        # 7. Impute remaining NaN (numeric) with median
        #    After label-encoding, categoricals are int (no NaN).
        #    Remaining NaN are in numeric columns (V1-V339, D1-D15, etc.).
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        medians = df[numeric_cols].median()
        df[numeric_cols] = df[numeric_cols].fillna(medians)

        # Fill any remaining NaN (e.g. object cols that slipped through)
        df = df.fillna(-1)

        # 8. Scale features (critical for SGDClassifier convergence)
        scaler = StandardScaler()
        self._X_full = scaler.fit_transform(df.values.astype(np.float64))
        self._y_full = y
        self._n_total = len(y)

        n_features = self._X_full.shape[1]
        fraud_rate = y.mean()

        print(f"  Preprocessed : {self._n_total:,} rows × {n_features} features")
        print(f"  Fraud rate   : {fraud_rate:.4f} ({y.sum():,} frauds)")
        print("  Data loading complete.\n")

