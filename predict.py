#!/usr/bin/env python3
"""
Reusable prediction command (assignment section "Reusable prediction command").

Usage:
   python predict.py --record "{\"plot_area_ha\":1.2,\"rainfall_mm\":81,\"soil_ph\":5.7,\"seed_kg\":210,\"distance_km\":14,\"arrival_hour\":9}"
    python predict.py --record '{"plot_area_ha":1.2}' --group AI-GXX

Loads the three models saved by run_all.py (regression, classification,
clustering), validates the incoming record against the documented schema,
and prints one JSON object with:
  regression_prediction, classification_prediction, classification_probability,
  cluster_label, group_code, model_version

Malformed or missing fields are rejected with a clear JSON error on stderr
and a non-zero exit code — never a raw Python traceback.
"""

import argparse
import json
import sys

import numpy as np

from src.utils import FEATURE_COLUMNS, MODEL_VERSION, ValidationError, validate_record


def parse_args():
    parser = argparse.ArgumentParser(description="Score one record with the trained models.")
    parser.add_argument("--record", required=True, help="A single JSON object matching the feature schema.")
    parser.add_argument("--models", default="models", help="Directory the trained models were saved into.")
    parser.add_argument("--group", default="AI-GXX", help="Group code to echo back in the output.")
    return parser.parse_args()


def load_models(models_dir: str):
    import joblib
    try:
        reg_bundle = joblib.load(f"{models_dir}/regression_model.joblib")
        clf_bundle = joblib.load(f"{models_dir}/classification_model.joblib")
        clu_bundle = joblib.load(f"{models_dir}/clustering_model.joblib")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find a trained model file ({e}). "
            "Run `python run_all.py --data ... --output ... --group ...` first."
        )
    return reg_bundle, clf_bundle, clu_bundle


def main():
    args = parse_args()

    # ---- Parse and validate the JSON record ----
    try:
        raw = json.loads(args.record)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}"}), file=sys.stderr)
        return 1

    try:
        cleaned = validate_record(raw)
    except ValidationError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

    # Build the feature vector in the exact column order the models expect
    X = np.array([[cleaned[col] for col in FEATURE_COLUMNS]], dtype=np.float64)

    # ---- Load models ----
    try:
        reg_bundle, clf_bundle, clu_bundle = load_models(args.models)
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

    # ---- Regression ----
    reg_model, reg_scaler = reg_bundle["model"], reg_bundle["scaler"]
    X_reg_scaled = reg_scaler.transform(X)
    regression_prediction = float(reg_model.predict(X_reg_scaled)[0])

    # ---- Classification ----
    clf_model, clf_scaler = clf_bundle["model"], clf_bundle["scaler"]
    X_clf_scaled = clf_scaler.transform(X)
    classification_prediction = int(clf_model.predict(X_clf_scaled)[0])
    classification_probability = float(clf_model.predict_proba(X_clf_scaled)[0][1])

    # ---- Clustering ----
    clu_model, clu_scaler = clu_bundle["model"], clu_bundle["scaler"]
    X_clu_scaled = clu_scaler.transform(X)
    cluster_label = int(clu_model.predict(X_clu_scaled)[0])

    result = {
        "regression_prediction_actual_yield_kg": round(regression_prediction, 3),
        "classification_prediction_dispatch_attention": classification_prediction,
        "classification_probability": round(classification_probability, 4),
        "cluster_label": cluster_label,
        "group_code": args.group,
        "model_version": MODEL_VERSION,
    }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
