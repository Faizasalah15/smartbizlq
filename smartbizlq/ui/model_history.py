"""Model history viewer displaying past model training runs, metrics, and parameters."""

import json
import logging
import os
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

from ..db.models import ModelRun
from ..db.session import SessionLocal


def _format_json_field(json_str: str | None) -> dict[str, Any]:
    """Parse a JSON string into a Python dict safely."""
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except Exception:
        return {"raw": str(json_str)}


def render_model_history_page() -> None:
    """Show past model runs from the ModelRun table with sortable views, metrics inspector, and artifact download."""
    st.header("Model Run History & Audit Log")
    st.markdown("Inspect, filter, and audit machine learning model training runs recorded in the database.")

    if SessionLocal is None or ModelRun is None:
        st.warning("Database session is not available. Please ensure database configuration is active.")
        return

    runs = []
    try:
        with SessionLocal() as db:
            runs = db.query(ModelRun).order_by(ModelRun.created_at.desc()).all()
    except Exception as e:
        st.error(f"Failed to query model run history: {e}")
        logger.exception("ModelRun query failure")
        return

    if not runs:
        st.info("No model runs recorded yet. Train a model in the 'ML Studio' to generate run records.")
        return

    # Extract records into structured list
    run_records = []
    for r in runs:
        metrics_dict = _format_json_field(r.metrics_json)
        params_dict = _format_json_field(r.params_json)

        # Create concise string summaries
        metric_summary = ", ".join(
            f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in list(metrics_dict.items())[:3]
        ) or "None"

        dataset_name = params_dict.get("dataset_name", "Unknown")

        run_records.append({
            "Run ID": r.id,
            "Model Name": r.model_name or "Unknown",
            "Version": r.version or "v1",
            "Dataset": dataset_name,
            "Metrics Summary": metric_summary,
            "Artifact Path": r.file_path or "N/A",
            "Trained At (UTC)": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "N/A",
            "_metrics_dict": metrics_dict,
            "_params_dict": params_dict,
            "_file_path": r.file_path,
        })

    history_df = pd.DataFrame(run_records)

    # Filtering Controls
    all_models = ["All"] + sorted(history_df["Model Name"].unique().tolist())
    filter_model = st.selectbox("Filter by model type", all_models, index=0, key="hist_filter_model")

    filtered_df = history_df
    if filter_model != "All":
        filtered_df = filtered_df[filtered_df["Model Name"] == filter_model]

    # Search bar for dataset or path
    search_term = st.text_input("Search dataset name or version", key="hist_search")
    if search_term:
        mask = (
            filtered_df["Dataset"].str.contains(search_term, case=False, na=False)
            | filtered_df["Version"].str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    st.markdown(f"**Found {len(filtered_df)} model runs:**")

    # Display clean table without internal dict columns
    display_cols = ["Run ID", "Model Name", "Version", "Dataset", "Metrics Summary", "Artifact Path", "Trained At (UTC)"]
    st.dataframe(filtered_df[display_cols], use_container_width=True)

    # Run Detail Inspector
    st.markdown("---")
    st.subheader("Model Run Inspector")

    run_ids = filtered_df["Run ID"].tolist()
    if not run_ids:
        st.info("No runs available for inspection under current filters.")
        return

    selected_id = st.selectbox("Select Run ID to inspect", run_ids, key="hist_inspect_id")
    selected_run = filtered_df[filtered_df["Run ID"] == selected_id].iloc[0]

    st.markdown(f"### Details for Run #{selected_id} — {selected_run['Model Name']} ({selected_run['Version']})")

    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.write(f"**Dataset:** `{selected_run['Dataset']}`")
        st.write(f"**Timestamp:** `{selected_run['Trained At (UTC)']}`")
    with col_meta2:
        st.write(f"**Model Artifact File:** `{selected_run['Artifact Path']}`")

    # Metrics Display
    metrics_data = selected_run["_metrics_dict"]
    if metrics_data:
        st.markdown("**Recorded Evaluation Metrics:**")
        # If metrics contain simple numeric floats/ints, display as metrics strip
        numeric_items = [(k, v) for k, v in metrics_data.items() if isinstance(v, (int, float))]
        if numeric_items:
            m_cols = st.columns(min(len(numeric_items), 5))
            for idx, (m_name, m_val) in enumerate(numeric_items[:5]):
                col_target = m_cols[idx % len(m_cols)]
                val_formatted = f"{m_val:.4f}" if isinstance(m_val, float) else f"{m_val:,}"
                col_target.metric(m_name, val_formatted)

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        with st.expander("Full Metrics Payload (JSON)", expanded=False):
            st.json(metrics_data)

    with exp_col2:
        with st.expander("Full Parameters Payload (JSON)", expanded=False):
            st.json(selected_run["_params_dict"])

    # Artifact Download Button if available on disk
    artifact_path = selected_run["_file_path"]
    if artifact_path and os.path.exists(artifact_path):
        try:
            with open(artifact_path, "rb") as af:
                artifact_bytes = af.read()
            st.download_button(
                label=f"Download Trained Model Artifact ({os.path.basename(artifact_path)})",
                data=artifact_bytes,
                file_name=os.path.basename(artifact_path),
                mime="application/octet-stream",
                key=f"dl_artifact_{selected_id}",
            )
        except Exception as dl_err:
            logger.warning("Could not read artifact file %s: %s", artifact_path, dl_err)
