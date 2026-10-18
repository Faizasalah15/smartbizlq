"""Dataset loading and session state management for SmartBizLQ.

Provides utilities for listing available datasets from disk and session,
loading DataFrames, caching columnar metadata in Streamlit session state,
and intelligent column matching.
"""

import logging
import os
import numpy as np
import pandas as pd
import streamlit as st

from .cleaning import safe_read_csv

logger = logging.getLogger(__name__)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_DIR = os.path.join(PROJECT_ROOT, "data", "sample")
DEMO_PREFIX = "Demo: "
SESSION_LABEL = "Session: Current upload (unsaved)"


def list_demo_datasets(demo_dir: str = DEMO_DIR) -> list[str]:
    """Return labels for the bundled demo CSV files in ``data/sample``."""
    if not os.path.isdir(demo_dir):
        return []
    return [DEMO_PREFIX + f for f in sorted(os.listdir(demo_dir)) if f.lower().endswith(".csv")]


def list_available_datasets(datasets_dir: str = "datasets") -> list[str]:
    """Check Streamlit session state and datasets directory for available datasets.

    Args:
        datasets_dir: Path to directory containing saved datasets. Defaults to 'datasets'.

    Returns:
        list[str]: List of dataset identifiers or filenames.
    """
    datasets: list[str] = []

    # Check for active session uploads
    try:
        if hasattr(st, "session_state"):
            if "uploaded_df" in st.session_state and st.session_state["uploaded_df"] is not None:
                if isinstance(st.session_state["uploaded_df"], pd.DataFrame) and not st.session_state["uploaded_df"].empty:
                    datasets.append("Session: Current upload (unsaved)")
    except Exception as e:
        logger.debug("Could not inspect session state for uploaded datasets: %s", e)

    # Check filesystem datasets directory
    if os.path.exists(datasets_dir):
        files = [
            f
            for f in sorted(os.listdir(datasets_dir))
            if os.path.isfile(os.path.join(datasets_dir, f))
        ]
        datasets.extend(files)

    datasets.extend(list_demo_datasets())
    return datasets


def load_dataset(selection: str, datasets_dir: str = "datasets") -> pd.DataFrame:
    """Load a dataset DataFrame given its selection identifier.

    If selection is 'Session: Current upload (unsaved)', returns a copy of
    `st.session_state['uploaded_df']`. Otherwise, loads the specified CSV file
    from `datasets_dir`.

    Args:
        selection: Dataset option label or file name.
        datasets_dir: Directory where dataset files reside.

    Returns:
        pd.DataFrame: Loaded DataFrame, or an empty DataFrame if loading fails.
    """
    if not selection:
        return pd.DataFrame()

    if selection.startswith(DEMO_PREFIX):
        return safe_read_csv(os.path.join(DEMO_DIR, selection[len(DEMO_PREFIX):]))

    if selection == SESSION_LABEL:
        try:
            if hasattr(st, "session_state") and "uploaded_df" in st.session_state:
                uploaded = st.session_state["uploaded_df"]
                if isinstance(uploaded, pd.DataFrame):
                    return uploaded.copy()
        except Exception as e:
            logger.warning("Error reading dataset from session state: %s", e)
        return pd.DataFrame()

    file_path = os.path.join(datasets_dir, selection)
    if not os.path.exists(file_path):
        if os.path.exists(selection):
            file_path = selection
        else:
            logger.error("Dataset not found at path: %s", file_path)
            return pd.DataFrame()

    return safe_read_csv(file_path)


def store_df_in_session(df: pd.DataFrame, name: str = "active_df") -> None:
    """Store a DataFrame and its detected column types in Streamlit session state.

    Extracts numeric, datetime candidate, and categorical column subsets into:
    - `{name}`: DataFrame copy
    - `{name}_all`: list of all column names
    - `{name}_numeric`: list of numeric column names
    - `{name}_cat`: list of categorical column names
    - `{name}_date`: list of datetime column candidates

    Args:
        df: Pandas DataFrame to store.
        name: Session state variable prefix (default 'active_df').
    """
    try:
        if not hasattr(st, "session_state"):
            logger.warning("Streamlit session_state is not accessible.")
            return
    except Exception as e:
        logger.warning("Failed to access Streamlit session state: %s", e)
        return

    if df is None or df.empty:
        st.session_state[name] = pd.DataFrame()
        st.session_state[f"{name}_numeric"] = []
        st.session_state[f"{name}_cat"] = []
        st.session_state[f"{name}_date"] = []
        st.session_state[f"{name}_all"] = []
        return

    st.session_state[name] = df.copy()
    all_cols: list[str] = list(df.columns)
    st.session_state[f"{name}_all"] = all_cols

    numeric_cols: list[str] = df.select_dtypes(include=[np.number]).columns.tolist()

    date_candidates: list[str] = []
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notnull().sum() > 0.3 * len(df):
                date_candidates.append(c)
        except Exception:
            pass

    cat_cols: list[str] = [
        c for c in all_cols if c not in numeric_cols and c not in date_candidates
    ]

    st.session_state[f"{name}_numeric"] = numeric_cols
    st.session_state[f"{name}_cat"] = cat_cols
    st.session_state[f"{name}_date"] = date_candidates


def find_column(cols: list[str], patterns: list[str]) -> str | None:
    """Find the first column name that matches any of the given pattern substrings.

    Search is case-insensitive and prioritizes patterns in the order given.

    Args:
        cols: Collection of column names to search.
        patterns: Patterns to search for within column names.

    Returns:
        str | None: Matching column name, or None if no match is found.
    """
    for p in patterns:
        for c in cols:
            try:
                if p.lower() in str(c).lower():
                    return str(c)
            except Exception:
                continue
    return None
