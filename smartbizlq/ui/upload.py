"""Dataset upload, inspection, automated data cleaning, and persistence UI."""

import logging
import os

import streamlit as st

logger = logging.getLogger(__name__)

from ..db.models import Dataset
from ..db.session import SessionLocal

from ..data.cleaning import clean_dataframe, safe_read_csv
from ..data.loaders import store_df_in_session


def render_upload_page() -> None:
    """Upload CSV dataset with preview, automated cleaning options, and persistence."""
    st.header("Upload CSV Dataset")
    st.markdown("Upload raw business data, inspect missingness, clean duplicates, and save to the platform.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="uploader")
    if not uploaded:
        st.info("Upload a CSV file to preview, clean, and activate it for dashboard and ML analysis.")
        return

    uploaded.seek(0)
    df = safe_read_csv(uploaded)

    if df is None or df.empty:
        st.error("The uploaded file could not be parsed or contains zero data rows.")
        return

    # Store raw uploaded dataset in session state
    st.session_state["uploaded_df"] = df.copy()
    st.session_state["uploaded_filename"] = uploaded.name

    st.subheader(f"Preview (First 20 Rows) — {uploaded.name}")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Columns", f"{len(df.columns):,}")
    c3.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    missing_series = df.isnull().sum()
    missing_cols = missing_series[missing_series > 0]
    if not missing_cols.empty:
        with st.expander("Columns with Missing Values"):
            st.write(missing_cols)

    # -------------------------------------------------------------
    # Cleaning Options
    # -------------------------------------------------------------
    st.markdown("### Automated Cleaning Options")
    cl_col1, cl_col2 = st.columns(2)

    with cl_col1:
        drop_empty_rows = st.checkbox("Drop rows that are entirely empty", value=True, key="clean_drop_rows")
        drop_empty_cols = st.checkbox("Drop columns that are entirely empty", value=True, key="clean_drop_cols")

    with cl_col2:
        drop_dups = st.checkbox("Drop duplicate rows", value=True, key="clean_drop_dups")
        row_thresh = st.slider("Row emptiness threshold (drop if > % empty)", 0.0, 0.9, 0.0, 0.05, key="clean_row_thresh")

    cleaned = clean_dataframe(
        df,
        drop_empty_rows=drop_empty_rows,
        drop_empty_cols=drop_empty_cols,
        row_thresh=row_thresh,
        drop_duplicates_flag=drop_dups,
    )

    st.subheader("Preview After Cleaning")
    st.dataframe(cleaned.head(20), use_container_width=True)
    st.write(f"Original records: **{len(df):,}** → Cleaned records: **{len(cleaned):,}** (Columns: **{len(cleaned.columns)}**)")

    # Action Buttons
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("Use Cleaned Dataset for Current Session", key="use_cleaned_session"):
            store_df_in_session(cleaned, name="active_df")
            st.session_state["uploaded_df"] = cleaned.copy()
            st.success(
                "Cleaned dataset saved into session memory! You can now select "
                "'Session: Current upload (unsaved)' in the Dashboard, Visuals Hub, or ML Studio."
            )

    with btn_col2:
        if st.button("Save Cleaned Dataset to Disk & Catalog", key="save_cleaned"):
            try:
                os.makedirs("datasets", exist_ok=True)
                target_path = os.path.join("datasets", uploaded.name)
                cleaned.to_csv(target_path, index=False)

                if SessionLocal is not None and Dataset is not None:
                    with SessionLocal() as db:
                        d_record = Dataset(
                            filename=uploaded.name,
                            schema_json=cleaned.dtypes.astype(str).to_json(),
                            rows_count=len(cleaned),
                        )
                        db.add(d_record)
                        db.commit()

                st.success(f"Saved cleaned dataset to: `{target_path}`")
                logger.info("Saved dataset %s with %d rows", target_path, len(cleaned))
            except Exception as e:
                st.error(f"Failed to persist dataset: {e}")
                logger.exception("Save dataset error")
