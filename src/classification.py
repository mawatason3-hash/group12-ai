from __future__ import annotations

from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import PipelineContext, write_json_atomic


def predict_attention(record_dict: dict[str, Any], pipeline: Pipeline, threshold: float = 0.5) -> tuple[int, float]:
    """Return the predicted attention label and probability for a single feature record."""
    feature_order = getattr(pipeline, "feature_order", None)
    if feature_order is None:
        feature_order = [key for key in record_dict if key not in {"record_id"}]

    row = [[float(record_dict.get(feature, np.nan)) for feature in feature_order]]
    probability = float(pipeline.predict_proba(row)[0, 1])
    label = int(probability >= float(threshold))
    return label, probability


def run(ctx: PipelineContext) -> dict[str, Any]:
    """Train a balanced logistic regression classifier and save the required metrics and pipeline artifacts.

    False negatives are costlier in dispatch triage, so recall is monitored closely in
    the saved metrics and cost interpretation note.
    """
    valid_mask = np.isfinite(ctx.y_clf) & np.isfinite(ctx.X).all(axis=1)
    X_valid = ctx.X[valid_mask]
    y_valid = ctx.y_clf[valid_mask].astype(int)

    if X_valid.size == 0 or y_valid.size == 0:
        raise ValueError("No valid rows remain for classification training.")

    unique_classes, class_counts = np.unique(y_valid, return_counts=True)
    if unique_classes.size < 2:
        raise ValueError("Classification target has only one class in the dataset; logistic regression requires both classes for training.")

    if np.min(class_counts) >= 2:
        stratified = True
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid,
            y_valid,
            test_size=0.2,
            random_state=ctx.seed,
            stratify=y_valid,
        )
        split_reason = "Stratified train/test split preserves class balance."
    else:
        stratified = False
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid,
            y_valid,
            test_size=0.2,
            random_state=ctx.seed,
        )
        split_reason = "Falling back to a non-stratified split because at least one class has fewer than two samples."

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=ctx.seed)),
    ])
    pipeline.feature_order = list(ctx.feature_names)
    pipeline.fit(X_train, y_train)

    threshold = 0.5
    probas = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probas >= threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    roc_auc = None
    if np.unique(y_test).size == 2:
        roc_auc = float(roc_auc_score(y_test, probas))

    class_names = ["no_attention", "attention"]
    class_counts_summary = {
        "no_attention": int(np.sum(y_valid == 0)),
        "attention": int(np.sum(y_valid == 1)),
    }

    feature_coefficients = {
        feature: float(coefficient)
        for feature, coefficient in zip(ctx.feature_names, pipeline.named_steps["clf"].coef_[0])
    }

    confusion_figure, confusion_axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted no_attention", "Predicted attention"],
        yticklabels=["Actual no_attention", "Actual attention"],
        cbar=False,
        ax=confusion_axis,
    )
    confusion_axis.set_xlabel("Predicted")
    confusion_axis.set_ylabel("Actual")
    confusion_axis.set_title("Dispatch Attention Confusion Matrix")
    confusion_figure.tight_layout()

    confusion_path = ctx.output_dir / "confusion_matrix.png"
    confusion_summary_path = ctx.output_dir / "classification_confusion_matrix.png"
    confusion_figure.savefig(confusion_path, dpi=150)
    confusion_figure.savefig(confusion_summary_path, dpi=150)
    plt.close(confusion_figure)

    model_json_path = ctx.models_dir / "classification_model.json"
    model_json_payload = {
        "feature_order": list(ctx.feature_names),
        "imputer_medians": pipeline.named_steps["imputer"].statistics_.tolist(),
        "scaler_mean": pipeline.named_steps["scaler"].mean_.tolist(),
        "scaler_scale": pipeline.named_steps["scaler"].scale_.tolist(),
        "coef": pipeline.named_steps["clf"].coef_.tolist(),
        "intercept": pipeline.named_steps["clf"].intercept_.tolist(),
        "classes": pipeline.named_steps["clf"].classes_.tolist(),
    }
    write_json_atomic(model_json_path, model_json_payload)
    joblib.dump(pipeline, ctx.models_dir / "classifier.joblib")

    metrics_payload = {
        "seed": ctx.seed,
        "threshold": float(threshold),
        "stratified": bool(stratified),
        "split_reason": split_reason,
        "class_counts": class_counts_summary,
        "labels": class_names,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "confusion_matrix": [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]],
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "feature_coefficients": feature_coefficients,
        "cost_interpretation": "In this dispatch scenario a false negative (a consignment needing attention that is missed) is more costly than a false positive (an extra check). Recall should be monitored closely.",
    }
    metrics_path = ctx.output_dir / "classification_metrics.json"
    summary_path = ctx.output_dir / "classification_summary.json"
    write_json_atomic(metrics_path, metrics_payload)
    write_json_atomic(summary_path, metrics_payload)

    return {
        "stage": "classification",
        "artifact_path": str(metrics_path),
        "status": "ok",
        "accuracy": float(accuracy),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
    }
