"""Unit tests for smartbizlq.data modules (cleaning and loaders)."""

import io
import numpy as np
import pandas as pd

from smartbizlq.data.cleaning import clean_dataframe, safe_read_csv
from smartbizlq.data.loaders import find_column, list_available_datasets, load_dataset


def test_safe_read_csv_comma_delimited():
    """Test reading a standard comma-separated CSV."""
    csv_data = "date,revenue,units\n2025-01-01,1000,50\n2025-01-02,1500,75\n"
    buffer = io.BytesIO(csv_data.encode("utf-8"))
    df = safe_read_csv(buffer)
    assert not df.empty
    assert list(df.columns) == ["date", "revenue", "units"]
    assert len(df) == 2


def test_safe_read_csv_tab_delimited():
    """Test reading a tab-delimited CSV buffer."""
    tsv_data = "date\trevenue\tunits\n2025-01-01\t1000\t50\n"
    buffer = io.BytesIO(tsv_data.encode("utf-8"))
    df = safe_read_csv(buffer)
    assert not df.empty
    assert list(df.columns) == ["date", "revenue", "units"]
    assert len(df) == 1


def test_clean_dataframe_drops_empty_rows_and_cols():
    """Test clean_dataframe drops entirely empty rows and columns."""
    raw_df = pd.DataFrame({
        "A": [1, np.nan, 3, np.nan],
        "B": ["x", "  ", "z", np.nan],
        "EmptyCol": [np.nan, " ", np.nan, np.nan],
    })
    cleaned = clean_dataframe(raw_df, drop_empty_rows=True, drop_empty_cols=True)
    assert "EmptyCol" not in cleaned.columns
    # Rows 2 and 4 contain only blanks/NaN once whitespace is treated as missing
    assert len(cleaned) == 2
    assert cleaned["A"].tolist() == [1.0, 3.0]


def test_clean_dataframe_duplicates():
    """Test clean_dataframe deduplication behavior."""
    raw_df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "val": ["a", "b", "b", "c"],
    })
    cleaned = clean_dataframe(raw_df, drop_duplicates_flag=True)
    assert len(cleaned) == 3
    assert list(cleaned["id"]) == [1, 2, 3]


def test_find_column():
    """Test matching column names case-insensitively using patterns."""
    cols = ["Transaction_Date", "Gross_Sales", "Total_Units", "SKU_Category"]
    assert find_column(cols, ["date", "time"]) == "Transaction_Date"
    assert find_column(cols, ["sales", "revenue"]) == "Gross_Sales"
    assert find_column(cols, ["units", "qty"]) == "Total_Units"
    assert find_column(cols, ["category", "item"]) == "SKU_Category"
    assert find_column(cols, ["nonexistent"]) is None


def test_list_and_load_dataset(tmp_path):
    """Test listing datasets from disk and loading them."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    sample_file = datasets_dir / "sample.csv"
    sample_file.write_text("colA,colB\n1,2\n3,4\n", encoding="utf-8")

    available = list_available_datasets(datasets_dir=str(datasets_dir))
    assert "sample.csv" in available

    loaded = load_dataset("sample.csv", datasets_dir=str(datasets_dir))
    assert not loaded.empty
    assert len(loaded) == 2


def test_coerce_numeric_columns_handles_blanks():
    from smartbizlq.data.cleaning import coerce_numeric_columns

    df = pd.DataFrame({"TotalCharges": ["29.85", "1,889.50", " ", "108.15"], "plan": ["a", "b", "c", "d"]})
    out = coerce_numeric_columns(df)
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])
    assert out["TotalCharges"].tolist()[:2] == [29.85, 1889.5]
    assert out["TotalCharges"].isna().sum() == 1
    assert out["plan"].dtype == object
