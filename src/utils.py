"""
Shared constants and small helper functions used across the pipeline.

Keeping the schema, random seed, and I/O helpers in one place means every
module (data pipeline, regression, classification, clustering, predict.py)
agrees on column names and reproducibility settings. This is what lets the
assessor swap in a hidden CSV with the same schema and have everything
still work without edits.
"""

import hashlib
import json
import os
import random

import numpy as np

# ---------------------------------------------------------------------
# Dataset schema (must match AI_A1_GXX.csv exactly — see assignment PDF)
# ---------------------------------------------------------------------
IDENTIFIER_COL = "record_id"
REGRESSION_TARGET = "actual_yield_kg"
CLASSIFICATION_TARGET = "dispatch_attention"

FEATURE_COLUMNS = [
    "plot_area_ha",
    "rainfall_mm",
    "soil_ph",
    "seed_kg",
    "distance_km",
    "arrival_hour",
]

ALL_REQUIRED_COLUMNS = (
    [IDENTIFIER_COL] + FEATURE_COLUMNS + [REGRESSION_TARGET, CLASSIFICATION_TARGET]
)

# Fixed random seed — used everywhere (NumPy, sklearn splits, KMeans init)
# so that re-running the pipeline on the same data gives the same result.
RANDOM_SEED = 42

MODEL_VERSION = "1.0.0"


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix every source of randomness we touch directly."""
    random.seed(seed)
    np.random.seed(seed)


def compute_sha256(filepath: str) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    """Let json.dump handle numpy types without crashing."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


class ValidationError(Exception):
    """Raised when an incoming record fails schema validation (used by predict.py)."""
    pass


def validate_record(record: dict) -> dict:
    """
    Validate a single prediction-time record against FEATURE_COLUMNS.

    Returns a dict of {column: float} in a fixed order on success.
    Raises ValidationError with a clear, specific message on failure —
    predict.py catches this and prints a clean JSON error instead of a
    Python traceback.
    """
    if not isinstance(record, dict):
        raise ValidationError("Input record must be a JSON object.")

    missing = [c for c in FEATURE_COLUMNS if c not in record]
    if missing:
        raise ValidationError(f"Missing required field(s): {', '.join(missing)}")

    cleaned = {}
    for col in FEATURE_COLUMNS:
        value = record[col]
        if isinstance(value, bool) or value is None:
            raise ValidationError(f"Field '{col}' must be numeric, got {value!r}.")
        try:
            cleaned[col] = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"Field '{col}' must be numeric, got {value!r}.")

    return cleaned
