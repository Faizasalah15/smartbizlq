"""Customer and transaction segmentation module using K-Means clustering.

This module provides an end-to-end unsupervised segmentation workflow:
- Feature scaling using StandardScaler to prevent metric scale distortion.
- Inertia elbow curve generation across candidate cluster counts k=2..max_k.
- K-Means partitioning for target cluster size.
- Silhouette score calculation for clustering cohesion and separation.
- Cluster profiling comparing cluster-level feature averages to population averages.
- Automated generation of plain-English executive cluster descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Dataclass holding complete clustering outputs, metrics, and profiles.

    Attributes
    ----------
    labels : np.ndarray
        Array of integer cluster labels assigned to each input observation.
    model : KMeans
        Fitted scikit-learn KMeans estimator for the selected cluster count.
    scaler : StandardScaler
        Fitted StandardScaler instance used to standardize input features.
    silhouette : float
        Silhouette score measuring cluster cohesion and separation (-1.0 to 1.0).
    elbow_data : list[tuple[int, float]]
        List of (k, inertia) tuples evaluated across k=2..max_k.
    profiles : pd.DataFrame
        Cluster profile DataFrame comparing cluster means vs overall population means.
    descriptions : list[str]
        Plain-English qualitative descriptions summarizing each cluster segment.
    """

    labels: np.ndarray
    model: KMeans
    scaler: StandardScaler
    silhouette: float
    elbow_data: list[tuple[int, float]]
    profiles: pd.DataFrame
    descriptions: list[str]
    silhouette_by_k: dict[int, float] = field(default_factory=dict)


def _build_cluster_description(
    cluster_id: int,
    size: int,
    pct_of_total: float,
    feature_means: dict[str, float],
    diff_pcts: dict[str, float],
) -> str:
    """Generate a human-readable, plain-English summary of a cluster segment."""
    # Identify features with the strongest deviations from baseline
    sorted_diffs = sorted(diff_pcts.items(), key=lambda item: abs(item[1]), reverse=True)

    traits: list[str] = []
    for feat, diff in sorted_diffs[:3]:
        mean_val = feature_means[feat]
        if diff >= 15.0:
            traits.append(f"high {feat} ({mean_val:.2f}, +{diff:.1f}% vs avg)")
        elif diff <= -15.0:
            traits.append(f"low {feat} ({mean_val:.2f}, {diff:.1f}% vs avg)")
        else:
            traits.append(f"moderate {feat} ({mean_val:.2f}, {diff:+.1f}% vs avg)")

    trait_summary = ", ".join(traits) if traits else "balanced metrics across all features"
    return f"Cluster {cluster_id} (N={size:,}, {pct_of_total:.1f}% of total): Characterized by {trait_summary}."


def run_clustering(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 3,
    max_k: int = 10,
) -> ClusterResult:
    """Execute K-Means clustering with standard scaling, elbow evaluation, and profiling.

    Steps:
    1. Select numeric features and fill missing values with 0.
    2. Normalize feature distributions using StandardScaler.
    3. Run K-Means across k=2..max_k to compute inertia for elbow plot analysis.
    4. Fit K-Means on normalized data for the chosen n_clusters.
    5. Compute overall silhouette coefficient.
    6. Generate cluster profiles (cluster-level mean vs population baseline).
    7. Generate plain-English descriptive summaries for each cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing observations to segment.
    feature_cols : list[str]
        List of numeric feature column names to use for segmentation.
    n_clusters : int, default=3
        Desired number of clusters for final model fitting.
    max_k : int, default=10
        Maximum number of clusters to evaluate for elbow curve analysis.

    Returns
    -------
    ClusterResult
        Dataclass containing cluster assignments, fitted model, scaler,
        silhouette score, elbow inertia data, profile DataFrame, and descriptions.

    Raises
    ------
    KeyError
        If any column in feature_cols is not present in df.
    ValueError
        If feature_cols is empty or dataset contains fewer records than n_clusters.
    """
    logger.info("Starting K-Means clustering on %d features with n_clusters=%d", len(feature_cols), n_clusters)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        msg = f"Missing feature column(s) for clustering: {missing}"
        logger.error(msg)
        raise KeyError(msg)

    if not feature_cols:
        msg = "At least one feature column must be specified for clustering."
        logger.error(msg)
        raise ValueError(msg)

    # 1. Select numeric features, coerce, and impute NaNs
    X_clean = df[feature_cols].copy()
    for col in feature_cols:
        X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce").fillna(0.0)

    n_samples = len(X_clean)
    if n_samples < n_clusters:
        msg = f"Insufficient sample size ({n_samples}) for {n_clusters} clusters. Need at least {n_clusters} rows."
        logger.error(msg)
        raise ValueError(msg)

    # 2. StandardScaler normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # 3. Evaluate candidate k=2..max_k for elbow curve
    effective_max_k = min(max_k, n_samples - 1)
    elbow_data: list[tuple[int, float]] = []
    silhouette_by_k: dict[int, float] = {}

    if effective_max_k >= 2:
        for k in range(2, effective_max_k + 1):
            km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
            k_labels = km_k.fit_predict(X_scaled)
            elbow_data.append((k, round(float(km_k.inertia_), 4)))
            if len(np.unique(k_labels)) > 1:
                silhouette_by_k[k] = round(
                    float(silhouette_score(X_scaled, k_labels, sample_size=min(n_samples, 5000), random_state=42)), 4
                )
        logger.debug("Computed elbow inertia curve for k=2..%d", effective_max_k)
    else:
        logger.warning("Dataset too small for multi-k elbow evaluation; skipping elbow sweep.")

    # 4. Fit K-Means on target n_clusters
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    # 5. Compute silhouette score
    if len(np.unique(labels)) > 1 and n_samples > n_clusters:
        try:
            sil_score = round(float(silhouette_score(X_scaled, labels)), 4)
        except Exception as sil_err:
            logger.warning("Silhouette calculation encountered error: %s", sil_err)
            sil_score = 0.0
    else:
        sil_score = 0.0

    # 6. Generate cluster profiles (cluster means vs overall population means)
    X_profile = X_clean.copy()
    X_profile["cluster"] = labels

    overall_means = X_clean.mean()

    profile_records: list[dict[str, float]] = []
    descriptions: list[str] = []

    for c in range(n_clusters):
        c_mask = labels == c
        c_size = int(np.sum(c_mask))
        c_pct = (c_size / n_samples) * 100.0 if n_samples > 0 else 0.0

        record: dict[str, float] = {
            "cluster": c,
            "size": float(c_size),
            "pct_of_total": round(c_pct, 2),
        }

        c_means: dict[str, float] = {}
        c_diffs: dict[str, float] = {}

        for col in feature_cols:
            c_mean = float(X_clean.loc[c_mask, col].mean()) if c_size > 0 else 0.0
            ov_mean = float(overall_means[col])
            c_means[col] = c_mean
            record[f"{col}_mean"] = round(c_mean, 2)

            # Percentage difference vs overall mean
            if abs(ov_mean) > 1e-8:
                diff_pct = ((c_mean - ov_mean) / abs(ov_mean)) * 100.0
            else:
                diff_pct = 0.0
            c_diffs[col] = diff_pct
            record[f"{col}_vs_avg_pct"] = round(diff_pct, 2)

        profile_records.append(record)

        # 7. Generate plain-English descriptive summary
        desc = _build_cluster_description(
            cluster_id=c,
            size=c_size,
            pct_of_total=c_pct,
            feature_means=c_means,
            diff_pcts=c_diffs,
        )
        descriptions.append(desc)

    profiles_df = pd.DataFrame(profile_records).set_index("cluster")

    logger.info(
        "K-Means clustering completed: k=%d, Silhouette=%.4f, Inertia=%.2f",
        n_clusters,
        sil_score,
        model.inertia_,
    )

    return ClusterResult(
        labels=labels,
        model=model,
        scaler=scaler,
        silhouette=sil_score,
        elbow_data=elbow_data,
        profiles=profiles_df,
        descriptions=descriptions,
        silhouette_by_k=silhouette_by_k,
    )
