from __future__ import annotations

from collections import Counter
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .common import PipelineContext, write_json_atomic


def predict_cluster(record_dict: dict[str, Any], model: dict[str, Any]) -> int:
    """Predict the cluster label for a single record using the saved imputer/scaler/KMeans bundle."""
    feature_order = model.get("feature_order", list(record_dict.keys()))
    row = np.asarray([float(record_dict.get(feature, np.nan)) for feature in feature_order], dtype=float).reshape(1, -1)
    imputer = model["imputer"]
    scaler = model["scaler"]
    kmeans = model["kmeans"]
    row_imputed = imputer.transform(row)
    row_scaled = scaler.transform(row_imputed)
    return int(kmeans.predict(row_scaled)[0])


def run(ctx: PipelineContext) -> dict[str, Any]:
    """Cluster the feature set using KMeans and pick the best k by silhouette score."""
    valid_mask = np.isfinite(ctx.X).all(axis=1)
    X_valid = ctx.X[valid_mask]
    if X_valid.shape[0] < 2:
        raise ValueError("At least two valid rows are required for clustering.")

    # The scaler is fit on all rows for this unsupervised step; there is no target leakage
    # because clustering is entirely exploratory and no test set exists for this stage.
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_valid)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    n_rows = X_valid.shape[0]
    if n_rows < 6:
        k_values = list(range(2, min(5, n_rows)))
        k_reason = f"Rows={n_rows} is below 6, so the search was restricted to k in {k_values}."
    else:
        k_values = list(range(2, 6))
        k_reason = "The dataset is large enough to evaluate k=2,3,4,5."

    silhouette_by_k: dict[int, float] = {}
    inertia_by_k: dict[int, float] = {}
    kmeans_by_k: dict[int, KMeans] = {}
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=ctx.seed)
        labels = model.fit_predict(X_scaled)
        silhouette_by_k[k] = float(silhouette_score(X_scaled, labels))
        inertia_by_k[k] = float(model.inertia_)
        kmeans_by_k[k] = model

    best_k = max(k_values, key=lambda k: silhouette_by_k[k])
    best_score = silhouette_by_k[best_k]
    runner_up_score = 0.0
    if len(k_values) > 1:
        ranked_scores = sorted(silhouette_by_k.values(), reverse=True)
        runner_up_score = float(ranked_scores[1]) if len(ranked_scores) > 1 else float(ranked_scores[0])
    margin = float(best_score - runner_up_score)
    note = "Separation is weak; cluster choice is marginal because the best score is within 0.05 of the runner-up."
    if margin >= 0.05:
        note = "Cluster separation is reasonably clear because the winning silhouette score is clearly ahead of the runner-up."

    best_model = kmeans_by_k[best_k]
    cluster_labels = best_model.predict(X_scaled)
    cluster_sizes = [int(count) for count in Counter(cluster_labels).values()]

    cluster_means_original: list[dict[str, Any]] = []
    for cluster_id in sorted(np.unique(cluster_labels).tolist()):
        cluster_mask = cluster_labels == cluster_id
        cluster_row = X_valid[cluster_mask]
        feature_summary = {"cluster": int(cluster_id)}
        for feature_name, value in zip(ctx.feature_names, cluster_row.mean(axis=0)):
            feature_summary[feature_name] = float(value)
        cluster_means_original.append(feature_summary)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cluster_id in sorted(np.unique(cluster_labels).tolist()):
        cluster_mask = cluster_labels == cluster_id
        ax.scatter(reduced[cluster_mask, 0], reduced[cluster_mask, 1], label=f"Cluster {int(cluster_id)}", alpha=0.8)
    ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% explained variance)")
    ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% explained variance)")
    ax.set_title(f"Cluster Projection (selected k={best_k})")
    ax.legend(title="Clusters")
    fig.tight_layout()
    plot_path = ctx.output_dir / "cluster_plot.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    model_bundle = {
        "feature_order": list(ctx.feature_names),
        "imputer": imputer,
        "scaler": scaler,
        "kmeans": best_model,
    }
    joblib.dump(model_bundle, ctx.models_dir / "clustering.joblib")
    write_json_atomic(ctx.models_dir / "clustering_model.json", {
        "feature_order": list(ctx.feature_names),
        "imputer_medians": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "selected_k": int(best_k),
        "silhouette_per_k": {str(key): float(value) for key, value in silhouette_by_k.items()},
        "inertia_per_k": {str(key): float(value) for key, value in inertia_by_k.items()},
        "cluster_labels": cluster_labels.tolist(),
    })

    cluster_assignments = np.full(len(ctx.df), -1, dtype=int)
    cluster_assignments[valid_mask] = cluster_labels
    clusters_df = pd.DataFrame({
        "record_id": ctx.ids.astype(str).tolist(),
        "cluster": cluster_assignments.astype(int).tolist(),
    })
    clusters_df.to_csv(ctx.output_dir / "clusters.csv", index=False)

    metrics_payload = {
        "seed": ctx.seed,
        "k_range": [int(value) for value in k_values],
        "silhouette_per_k": {str(key): float(value) for key, value in silhouette_by_k.items()},
        "inertia_per_k": {str(key): float(value) for key, value in inertia_by_k.items()},
        "selected_k": int(best_k),
        "justification": {
            "best_score": float(best_score),
            "runner_up_score": float(runner_up_score),
            "margin": float(margin),
            "note": note,
        },
        "cluster_sizes": cluster_sizes,
        "cluster_means_original": cluster_means_original,
        "interpretation_note": "Clusters are statistical groupings of operating profiles, not verified real-world categories.",
        "search_note": k_reason,
    }
    metrics_path = ctx.output_dir / "clustering_metrics.json"
    summary_path = ctx.output_dir / "clustering_summary.json"
    write_json_atomic(metrics_path, metrics_payload)
    write_json_atomic(summary_path, metrics_payload)

    return {
        "stage": "clustering",
        "artifact_path": str(metrics_path),
        "status": "ok",
        "selected_k": int(best_k),
        "silhouette_score": float(best_score),
    }
