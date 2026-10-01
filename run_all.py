#!/usr/bin/env python3
"""
Musanze Cooperative Harvest and Dispatch Decision Lab — full pipeline.
"""

import argparse
import sys

from src.classification import run_classification
from src.clustering import run_clustering
from src.data_pipeline import build_data_report, clean_dataset, load_and_validate, vectorize
from src.regression import run_regression
from src.utils import RANDOM_SEED, compute_sha256, ensure_dir, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full decision-lab pipeline.")
    parser.add_argument("--data", required=True, help="Path to the group CSV file.")
    parser.add_argument("--output", required=True, help="Directory to write artifacts into.")
    parser.add_argument("--group", required=True, help="Group code, e.g. AI-GXX.")
    parser.add_argument("--models", default="models", help="Directory to save trained models into.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Gradient descent learning rate.")
    parser.add_argument("--iterations", type=int, default=1000, help="Gradient descent iteration count.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output)
    ensure_dir(args.models)

    # Step 1: data pipeline
    df_raw = load_and_validate(args.data)
    df_clean = clean_dataset(df_raw)
    fingerprint = compute_sha256(args.data)
    data_report = build_data_report(df_raw, df_clean, args.data, args.group)
    save_json(data_report, f"{args.output}/data_report.json")
    record_ids, X, y_reg, y_clf = vectorize(df_clean)

    # Step 2: regression
    reg_metrics = run_regression(
        X, y_reg, args.output, args.models,
        learning_rate=args.learning_rate, n_iterations=args.iterations, seed=args.seed,
    )

    # Step 3: classification
    clf_metrics = run_classification(X, y_clf, args.output, args.models, seed=args.seed)

    # Step 4: clustering
    cluster_metrics = run_clustering(record_ids, X, args.output, args.models, seed=args.seed)
    best_k = cluster_metrics["selected_k"]
    best_score = cluster_metrics["silhouette_scores_by_k"][best_k]

    # Clean column output module by module
    print("\nData Pipeline")
    print(f"  {'Raw Rows':<20} {len(df_raw)}")
    print(f"  {'Clean Rows':<20} {len(df_clean)}")
    print(f"  {'Fingerprint':<20} {fingerprint}")
    print(f"  {'Report Saved':<20} {args.output}/data_report.json")

    print("\nRegression")
    print(f"  {'Test MAE':<20} {reg_metrics['test_mae']:.3f}")
    print(f"  {'Test RMSE':<20} {reg_metrics['test_rmse']:.3f}")
    print(f"  {'Test R2':<20} {reg_metrics['test_r_squared']:.3f}")

    print("\nClassification")
    print(f"  {'Accuracy':<20} {clf_metrics['accuracy']:.3f}")
    print(f"  {'Precision':<20} {clf_metrics['precision']:.3f}")
    print(f"  {'Recall':<20} {clf_metrics['recall']:.3f}")
    print(f"  {'F1 Score':<20} {clf_metrics['f1_score']:.3f}")

    print("\nClustering")
    print(f"  {'Selected k':<20} {best_k}")
    print(f"  {'Silhouette Score':<20} {best_score:.3f}")

    print(f"\nDone. Artifacts written to: {args.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())