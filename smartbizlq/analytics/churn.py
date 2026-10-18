"""Customer churn classification and feature importance module.

This module provides an end-to-end machine learning pipeline for churn prediction:
- Automatic column type inspection (categorical vs numeric).
- Scikit-learn Pipeline with ColumnTransformer and OneHotEncoder.
- Class-weight balancing for skewed churn distributions.
- Stratified train/test splitting and Stratified K-Fold cross-validation.
- Out-of-sample test evaluation (Accuracy, Precision, Recall, F1, ROC-AUC).
- Permutation feature importance on the original domain features.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .kpis import compute_classification_metrics

logger = logging.getLogger(__name__)


@dataclass
class ChurnResult:
    """Dataclass holding complete churn classification artifacts and evaluation metrics.

    Attributes
    ----------
    model : Pipeline
        Trained scikit-learn Pipeline containing preprocessing and classifier.
    metrics : dict[str, float]
        Out-of-sample evaluation metrics: accuracy, precision, recall, f1, roc_auc.
    cv_scores : dict[str, Any]
        Cross-validation fold scores, mean, and standard deviation.
    confusion_matrix : np.ndarray
        Test set confusion matrix array.
    feature_importance : list[tuple[str, float]]
        Permutation importance per original feature, sorted in descending order.
    X_test : pd.DataFrame
        Held-out test features for validation and explainability visuals.
    y_test : pd.Series
        Held-out true target labels.
    y_pred : np.ndarray
        Model discrete class predictions on X_test.
    y_prob : np.ndarray
        Model predicted churn probabilities on X_test.
    """

    model: Pipeline
    metrics: dict[str, float]
    cv_scores: dict[str, Any]
    confusion_matrix: np.ndarray
    feature_importance: list[tuple[str, float]]
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_prob: np.ndarray


_POSITIVE = {"1", "yes", "y", "true", "t", "churn", "churned"}
_NEGATIVE = {"0", "no", "n", "false", "f", "stay", "retained"}


def _to_binary(series: pd.Series) -> pd.Series:
    """Map a churn label column to 1 (churned) / 0 (stayed); unknown values become NaN."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    return text.map(lambda v: 1.0 if v in _POSITIVE else (0.0 if v in _NEGATIVE else np.nan))


def train_churn_model(
    df: pd.DataFrame,
    churn_col: str,
    feature_cols: list[str],
    n_estimators: int = 100,
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> ChurnResult:
    """Train a churn classifier using a modular, leak-free scikit-learn Pipeline.

    Execution steps:
    1. Validate input columns and segregate numeric vs categorical features.
    2. Build ColumnTransformer (OneHotEncoder for categorical, passthrough for numeric).
    3. Build Pipeline wrapping ColumnTransformer -> RandomForestClassifier.
    4. Configure class_weight='balanced' to handle typical churn class imbalances.
    5. Perform stratified train/test split.
    6. Execute stratified k-fold cross-validation on training data.
    7. Fit the full pipeline on training data.
    8. Predict discrete classes and continuous probabilities on held-out test data.
    9. Compute classification metrics (ROC-AUC, Precision, Recall, F1, Accuracy).
    10. Compute test confusion matrix.
    11. Compute permutation feature importance on original feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features and binary churn target.
    churn_col : str
        Target column name representing customer churn (binary 0/1 or boolean).
    feature_cols : list[str]
        List of predictor feature column names.
    n_estimators : int, default=100
        Number of trees in the Random Forest ensemble.
    test_size : float, default=0.2
        Fraction of data reserved for out-of-sample testing.
    cv_folds : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    ChurnResult
        Complete container with fitted Pipeline, evaluation metrics, CV scores,
        confusion matrix, feature importances, and test split data.

    Raises
    ------
    KeyError
        If churn_col or any feature_cols are not found in df.
    ValueError
        If target column has fewer than 2 classes or feature set is empty.
    """
    logger.info("Initiating churn model training on target='%s' with %d features", churn_col, len(feature_cols))

    if churn_col not in df.columns:
        msg = f"Churn target column '{churn_col}' not found in DataFrame."
        logger.error(msg)
        raise KeyError(msg)

    missing_feats = [c for c in feature_cols if c not in df.columns]
    if missing_feats:
        msg = f"Missing feature column(s): {missing_feats}"
        logger.error(msg)
        raise KeyError(msg)

    if not feature_cols:
        msg = "Feature column list must contain at least one valid feature."
        logger.error(msg)
        raise ValueError(msg)

    # 1. Prepare clean data partition
    clean_cols = [churn_col] + feature_cols
    df_clean = df[clean_cols].dropna(subset=[churn_col]).copy()

    # Convert churn target to 0/1. Accepts numbers, booleans and common text labels.
    df_clean[churn_col] = _to_binary(df_clean[churn_col])
    df_clean = df_clean.dropna(subset=[churn_col])
    df_clean[churn_col] = df_clean[churn_col].astype(int)

    unique_targets = df_clean[churn_col].unique()
    if len(unique_targets) < 2:
        msg = f"Churn target requires at least 2 distinct classes; found {unique_targets}."
        logger.error(msg)
        raise ValueError(msg)

    # Identify numeric vs categorical features
    numeric_features: list[str] = []
    categorical_features: list[str] = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            numeric_features.append(col)
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0)
        else:
            categorical_features.append(col)
            df_clean[col] = df_clean[col].astype(str).fillna("missing")

    logger.debug(
        "Feature segregation: %d numeric features, %d categorical features",
        len(numeric_features),
        len(categorical_features),
    )

    # 2. Build ColumnTransformer
    transformers = []
    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )
    if numeric_features:
        transformers.append(
            ("num", "passthrough", numeric_features)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # 3. Build Pipeline: ColumnTransformer -> RandomForestClassifier
    # 4. Use class_weight='balanced' for class imbalance
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 5. Stratified train/test split
    X = df_clean[feature_cols].copy()
    y = df_clean[churn_col].copy()

    # Guard against stratify failure if class counts are very low
    class_counts = y.value_counts()
    stratify_target = y if class_counts.min() >= 2 else None
    if stratify_target is None:
        logger.warning("Minority class has fewer than 2 samples; disabling stratification in train_test_split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_target,
    )

    # 6. Stratified k-fold cross-validation on training partition
    min_train_class = int(y_train.value_counts().min()) if y_train.nunique() > 1 else 1
    actual_folds = min(cv_folds, min_train_class)

    if actual_folds >= 2:
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        try:
            cv_raw = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="roc_auc")
            cv_scores = {
                "folds": [round(float(s), 4) for s in cv_raw],
                "mean": round(float(np.mean(cv_raw)), 4),
                "std": round(float(np.std(cv_raw)), 4),
            }
        except Exception as cv_exc:
            logger.warning("Cross-validation failed with scoring='roc_auc': %s. Falling back to accuracy.", cv_exc)
            cv_raw = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="accuracy")
            cv_scores = {
                "folds": [round(float(s), 4) for s in cv_raw],
                "mean": round(float(np.mean(cv_raw)), 4),
                "std": round(float(np.std(cv_raw)), 4),
            }
    else:
        logger.warning("Training set has insufficient class representation for %d-fold CV.", cv_folds)
        cv_scores = {"folds": [], "mean": 0.0, "std": 0.0}

    # 7. Fit pipeline on full training data
    pipeline.fit(X_train, y_train)

    # 8. Predict and predict_proba on held-out test data
    y_pred = pipeline.predict(X_test)
    y_prob_raw = pipeline.predict_proba(X_test)

    # For binary classification, extract positive class probabilities
    if y_prob_raw.ndim == 2 and y_prob_raw.shape[1] == 2:
        y_prob = y_prob_raw[:, 1]
    else:
        y_prob = y_prob_raw

    # 9. Compute evaluation metrics
    metrics = compute_classification_metrics(y_test, y_pred, y_prob=y_prob)

    # 10. Compute confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)

    # 11. Compute permutation feature importance on test partition
    scoring_metric = "roc_auc" if (len(np.unique(y_test)) == 2 and y_test.value_counts().min() >= 1) else "accuracy"
    try:
        perm_res = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring=scoring_metric,
            n_jobs=-1,
        )
        feature_importance = [
            (col, round(float(imp), 4))
            for col, imp in zip(feature_cols, perm_res.importances_mean)
        ]
        feature_importance.sort(key=lambda item: item[1], reverse=True)
    except Exception as perm_err:
        logger.warning("Permutation importance calculation fallback: %s", perm_err)
        feature_importance = [(col, 0.0) for col in feature_cols]

    logger.info(
        "Churn model training finished. Test Metrics: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f, ROC-AUC=%.4f",
        metrics.get("accuracy", 0.0),
        metrics.get("precision", 0.0),
        metrics.get("recall", 0.0),
        metrics.get("f1", 0.0),
        metrics.get("roc_auc", 0.0),
    )

    return ChurnResult(
        model=pipeline,
        metrics=metrics,
        cv_scores=cv_scores,
        confusion_matrix=conf_mat,
        feature_importance=feature_importance,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )
