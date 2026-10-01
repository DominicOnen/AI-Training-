"""
Data and vectorization pipeline (assignment section 1).

Loads the lecturer-issued CSV, validates it against the documented schema,
reports missing/duplicate values, separates identifiers and targets from
model features, and produces a NumPy feature matrix. Also computes the
SHA-256 fingerprint of the raw file, which must be printed by run_all.py
and must match what's reported in data_report.json.
"""

import os

import numpy as np
import pandas as pd

from src.utils import (
    ALL_REQUIRED_COLUMNS,
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    IDENTIFIER_COL,
    REGRESSION_TARGET,
    compute_sha256,
)


class SchemaError(Exception):
    """Raised when the CSV doesn't match the documented schema."""
    pass


def load_and_validate(csv_path: str) -> pd.DataFrame:
    """Load the CSV and check that every required column is present."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_cols = [c for c in ALL_REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise SchemaError(
            "Dataset is missing required column(s): "
            f"{missing_cols}. Expected columns: {ALL_REQUIRED_COLUMNS}"
        )

    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows and rows missing a feature or target value.
    (Counts of what was removed are captured separately in build_data_report
    so nothing is silently lost from the audit trail.)
    """
    df_clean = df.drop_duplicates().copy()
    required_for_rows = FEATURE_COLUMNS + [REGRESSION_TARGET, CLASSIFICATION_TARGET]
    df_clean = df_clean.dropna(subset=required_for_rows)
    return df_clean.reset_index(drop=True)


def build_data_report(
    df_raw: pd.DataFrame, df_clean: pd.DataFrame, csv_path: str, group_code: str
) -> dict:
    """Assemble the dict that gets saved as artifacts/data_report.json."""
    missing_counts = df_raw[ALL_REQUIRED_COLUMNS].isna().sum()
    duplicate_count = int(df_raw.duplicated().sum())

    descriptive_stats = (
        df_clean[FEATURE_COLUMNS + [REGRESSION_TARGET]].describe().to_dict()
    )

    report = {
        "group_code": group_code,
        "dataset_path": csv_path,
        "dataset_sha256": compute_sha256(csv_path),
        "row_count_raw": int(len(df_raw)),
        "row_count_clean": int(len(df_clean)),
        "duplicate_rows_removed": duplicate_count,
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "identifier_column": IDENTIFIER_COL,
        "regression_target": REGRESSION_TARGET,
        "classification_target": CLASSIFICATION_TARGET,
        "missing_values_by_column": {
            k: int(v) for k, v in missing_counts.items()
        },
        "descriptive_statistics": descriptive_stats,
    }
    return report


def vectorize(df: pd.DataFrame):
    """
    Split a cleaned DataFrame into:
      - record_ids   : identifiers, never used as model input
      - X             : NumPy feature matrix (float64), columns = FEATURE_COLUMNS
      - y_reg         : regression target vector
      - y_clf         : classification target vector (0/1 ints)
    """
    record_ids = df[IDENTIFIER_COL].to_numpy()
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_reg = df[REGRESSION_TARGET].to_numpy(dtype=np.float64)
    y_clf = df[CLASSIFICATION_TARGET].to_numpy(dtype=int)
    return record_ids, X, y_reg, y_clf
