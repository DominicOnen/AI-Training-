"""
Clustering (assignment section 4).

Standardizes input features only (never the regression or classification
targets — clustering must not "see" the answers), evaluates k from 2 to 5
using silhouette score, picks the best k, and labels every record. Results
are described cautiously: clusters are operating-profile groupings, not
verified real-world categories.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils import FEATURE_COLUMNS, IDENTIFIER_COL, RANDOM_SEED, ensure_dir, save_json

CAUTION_NOTE = (
    "These clusters describe groups of collection points with similar "
    "operating profiles (area, rainfall, soil pH, seed quantity, distance, "
    "arrival timing) as measured in this dataset. They are a descriptive "
    "grouping, not a verified real-world category — a cluster should be "
    "treated as a starting point for operational review, not a label of "
    "cooperative performance or farm quality."
)


def run_clustering(record_ids: np.ndarray, X: np.ndarray, output_dir: str,
                    models_dir: str, seed: int = RANDOM_SEED) -> dict:
    start = time.time()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k_scores = {}
    fitted_models = {}
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        k_scores[k] = float(score)
        fitted_models[k] = (km, labels)

    best_k = max(k_scores, key=k_scores.get)
    best_model, best_labels = fitted_models[best_k]

    metrics = {
        "random_seed": seed,
        "feature_columns_used": FEATURE_COLUMNS,
        "k_candidates_evaluated": list(range(2, 6)),
        "silhouette_scores_by_k": k_scores,
        "selected_k": best_k,
        "selection_rule": "k with the highest silhouette score among k=2..5",
        "cluster_sizes": {
            int(c): int(n) for c, n in zip(*np.unique(best_labels, return_counts=True))
        },
        "caution": CAUTION_NOTE,
        "runtime_seconds": round(time.time() - start, 4),
    }

    ensure_dir(output_dir)
    save_json(metrics, f"{output_dir}/clustering_metrics.json")

    # Per-record cluster labels
    clusters_df = pd.DataFrame({
        IDENTIFIER_COL: record_ids,
        "cluster_label": best_labels,
    })
    clusters_df.to_csv(f"{output_dir}/clusters.csv", index=False)

    # 2D visualization via PCA (for plotting only — clustering itself used
    # all standardized features, not these 2 components)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6.5, 5.5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=best_labels, cmap="viridis", s=40, alpha=0.8)
    plt.title(f"Collection Point Clusters (k={best_k}, PCA projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Cluster label")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_plot.png", dpi=150)
    plt.close()

    ensure_dir(models_dir)
    import joblib
    joblib.dump(
        {"model": best_model, "scaler": scaler, "k": best_k},
        f"{models_dir}/clustering_model.joblib",
    )

    return metrics
