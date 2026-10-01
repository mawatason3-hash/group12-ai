from __future__ import annotations

from collections import Counter
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .common import PipelineContext, write_json_atomic


def run(ctx: PipelineContext) -> dict[str, Any]:
    """Cluster the normalized field data and select the best cluster count by silhouette score."""
    valid_mask = np.isfinite(ctx.X).all(axis=1)
    X_valid = ctx.X[valid_mask]
    if X_valid.shape[0] < 2:
        raise ValueError("At least two valid rows are required for clustering.")

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_valid)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    max_clusters = min(6, len(X_valid) // 2)
    if max_clusters < 2:
        max_clusters = 2

    best_k = 2
    best_score = -1.0
    best_model: KMeans | None = None
    silhouette_by_k: dict[int, float] = {}

    for n_clusters in range(2, max_clusters + 1):
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=ctx.seed)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_by_k[n_clusters] = float(score)
        if score > best_score:
            best_score = float(score)
            best_k = n_clusters
            best_model = model

    if best_model is None:
        raise ValueError("Unable to fit a clustering model.")

    labels = best_model.predict(X_scaled)
    cluster_sizes = dict(sorted(Counter(labels.tolist()).items()))

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)
    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis", s=18)
    axis.set_title("Cluster Projection")
    axis.set_xlabel("PCA 1")
    axis.set_ylabel("PCA 2")
    figure.colorbar(scatter, ax=axis)
    figure.tight_layout()
    cluster_plot_path = ctx.output_dir / "clustering_projection.png"
    plt.savefig(cluster_plot_path, dpi=150)
    plt.close(figure)

    model_payload = {
        "feature_order": list(ctx.feature_names),
        "imputer_medians": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "best_k": int(best_k),
        "silhouette_score": float(best_score),
        "cluster_centers": best_model.cluster_centers_.tolist(),
        "labels": labels.tolist(),
    }
    model_path = ctx.models_dir / "clustering_model.json"
    write_json_atomic(model_path, model_payload)

    payload = {
        "stage": "clustering",
        "group_code": ctx.group_code,
        "model_version": ctx.model_version,
        "feature_names": ctx.feature_names,
        "rows": int(len(ctx.df)),
        "valid_rows": int(len(X_valid)),
        "best_k": int(best_k),
        "silhouette_score": float(best_score),
        "silhouette_by_k": {str(key): float(value) for key, value in silhouette_by_k.items()},
        "cluster_sizes": {str(key): int(value) for key, value in cluster_sizes.items()},
        "status": "ok",
    }
    artifact_path = ctx.output_dir / "clustering_summary.json"
    write_json_atomic(artifact_path, payload)
    return {
        "stage": "clustering",
        "artifact_path": str(artifact_path),
        "status": "ok",
        "best_k": int(best_k),
        "silhouette_score": float(best_score),
    }
