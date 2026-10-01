from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .common import PipelineContext, write_json_atomic


def run(ctx: PipelineContext) -> dict[str, Any]:
    """Train a binary classifier for dispatch_attention and persist the metrics and model snapshot."""
    valid_mask = np.isfinite(ctx.y_clf) & np.isfinite(ctx.X).all(axis=1)
    X_valid = ctx.X[valid_mask]
    y_valid = ctx.y_clf[valid_mask].astype(int)
    if X_valid.size == 0 or y_valid.size == 0:
        raise ValueError("No valid rows remain for classification training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X_valid,
        y_valid,
        test_size=0.2,
        random_state=ctx.seed,
        stratify=y_valid,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    model = LogisticRegression(max_iter=5000, random_state=ctx.seed, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(cm, cmap="Blues")
    axis.set_title("Dispatch Classification Confusion Matrix")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["No dispatch", "Dispatch"])
    axis.set_yticklabels(["No dispatch", "Dispatch"])
    for row_index in range(cm.shape[0]):
        for col_index in range(cm.shape[1]):
            axis.text(col_index, row_index, int(cm[row_index, col_index]), ha="center", va="center", color="black")
    figure.tight_layout()
    confusion_path = ctx.output_dir / "classification_confusion_matrix.png"
    plt.savefig(confusion_path, dpi=150)
    plt.close(figure)

    model_payload = {
        "feature_order": list(ctx.feature_names),
        "imputer_medians": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "classes": model.classes_.tolist(),
    }
    model_path = ctx.models_dir / "classification_model.json"
    write_json_atomic(model_path, model_payload)

    payload = {
        "stage": "classification",
        "group_code": ctx.group_code,
        "model_version": ctx.model_version,
        "feature_names": ctx.feature_names,
        "rows": int(len(ctx.df)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "status": "ok",
    }
    artifact_path = ctx.output_dir / "classification_summary.json"
    write_json_atomic(artifact_path, payload)
    return {
        "stage": "classification",
        "artifact_path": str(artifact_path),
        "status": "ok",
        "accuracy": float(accuracy),
        "f1": float(f1),
    }
