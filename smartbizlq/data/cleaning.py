"""Data reading and cleaning routines for SmartBizLQ.

Provides robust CSV parsing with delimiter detection, string sanitization,
type inference, and DataFrame cleaning utilities.
"""

import csv
import io
import logging
import math
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def safe_read_csv(path_or_buffer: Any) -> pd.DataFrame:
    """Read a CSV dataset from a file path or in-memory byte buffer safely.

    Detects common delimiters (comma, tab, semicolon, pipe), strips trailing
    whitespace from column headers, sanitizes multiline strings, and attempts
    numeric type conversion for stringified numbers.

    Args:
        path_or_buffer: File system path (str/Path) or file-like buffer (e.g. BytesIO, UploadedFile).

    Returns:
        pd.DataFrame: Parsed and pre-cleaned DataFrame, or an empty DataFrame on error.
    """
    try:
        if isinstance(path_or_buffer, (str, Path)):
            with open(path_or_buffer, "rb") as f:
                raw_bytes = f.read()
            raw: str = raw_bytes.decode("utf-8", errors="replace")
            sample: str = raw[:10000]
            sep: str | None = None
            if sample.count("\t") > sample.count(","):
                sep = "\t"
            elif sample.count(";") > sample.count(","):
                sep = ";"
            else:
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
                    sep = dialect.delimiter
                except Exception:
                    sep = ","
            df = pd.read_csv(io.StringIO(raw), sep=sep, engine="python")
        else:
            try:
                path_or_buffer.seek(0)
            except Exception:
                pass
            raw_bytes = path_or_buffer.read()
            raw = (
                raw_bytes.decode("utf-8", errors="replace")
                if isinstance(raw_bytes, (bytes, bytearray))
                else str(raw_bytes)
            )
            sample = raw[:10000]
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
                sep = dialect.delimiter
            except Exception:
                if "\t" in sample and sample.count("\t") > sample.count(","):
                    sep = "\t"
                elif ";" in sample and sample.count(";") > sample.count(","):
                    sep = ";"
                else:
                    sep = ","
            df = pd.read_csv(io.StringIO(raw), sep=sep, engine="python")

        # Strip whitespace from column names
        df.columns = [str(c).strip() for c in df.columns]

        # Sanitize object/string columns
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for c in obj_cols:
            df[c] = df[c].astype(str).replace("nan", np.nan)
            df[c] = df[c].str.replace(r"[\r\n]+", " ", regex=True)
            df[c] = df[c].str.replace("\t", " ", regex=False)
            df[c] = df[c].str.strip()

        # Infer numeric columns that were parsed as strings
        for c in df.columns:
            if df[c].dtype == object:
                sample_vals = df[c].dropna().astype(str).head(200)
                if len(sample_vals) == 0:
                    continue
                sample_clean = sample_vals.str.replace(",", "", regex=False)
                is_num = sample_clean.str.match(r"^-?\d+(\.\d+)?$")
                if is_num.sum() >= max(1, int(0.6 * len(sample_clean))):
                    df[c] = pd.to_numeric(
                        df[c].astype(str).str.replace(r"[^\d\.\-]", "", regex=True).replace("", np.nan),
                        errors="coerce",
                    )
        return df

    except Exception as e:
        logger.error("Failed to read/clean CSV: %s", e, exc_info=True)
        return pd.DataFrame()


def clean_dataframe(
    df: pd.DataFrame,
    drop_empty_rows: bool = True,
    drop_empty_cols: bool = True,
    row_thresh: float = 0.0,
    drop_duplicates_flag: bool = True,
    dup_subset: list[str] | None = None,
    dup_keep: str = "first",
) -> pd.DataFrame:
    """Clean a pandas DataFrame with configurable filtering rules.

    Args:
        df: Input DataFrame to clean.
        drop_empty_rows: Whether to drop rows that are all or mostly empty.
        drop_empty_cols: Whether to drop columns that are entirely empty or whitespace.
        row_thresh: Fraction of empty columns required to drop a row (0.0 to 1.0).
        drop_duplicates_flag: Whether to eliminate duplicate rows.
        dup_subset: Specific subset of column names for duplicate identification.
        dup_keep: Which duplicate occurrence to retain ('first', 'last', or False).

    Returns:
        pd.DataFrame: Cleaned DataFrame with reset integer index.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df2 = df.copy()

    # Treat blank / whitespace-only text as missing, then drop columns with no data at all
    df2 = df2.replace(r"^\s*$", np.nan, regex=True).infer_objects()
    if drop_empty_cols:
        df2 = df2.loc[:, df2.notna().any(axis=0)]

    # Drop empty or mostly empty rows
    if drop_empty_rows:
        if row_thresh <= 0:
            df2 = df2.dropna(how="all")
        else:
            thresh = int(math.ceil((1.0 - row_thresh) * df2.shape[1]))
            if thresh < 1:
                thresh = 1
            df2 = df2.dropna(thresh=thresh)

    # Drop duplicate rows
    if drop_duplicates_flag:
        try:
            if dup_subset:
                df2 = df2.drop_duplicates(subset=dup_subset, keep=dup_keep)
            else:
                df2 = df2.drop_duplicates(keep=dup_keep)
        except Exception as e:
            logger.warning("Error during duplicate removal; retrying without subset: %s", e)
            df2 = df2.drop_duplicates(keep=dup_keep)

    return df2.reset_index(drop=True)


def coerce_numeric_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Convert text columns that are almost entirely numbers into numeric columns.

    For example, a 'TotalCharges' column stored as text because a few rows are blank.
    Values that cannot be parsed become NaN.

    Args:
        df: Input DataFrame (not modified).
        threshold: Minimum share of non-blank values that must parse as numbers.

    Returns:
        A copy of ``df`` with qualifying columns converted.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        text = out[col].astype(str).str.strip()
        non_blank = text[(text != "") & (text.str.lower() != "nan")]
        if non_blank.empty:
            continue
        parsed = pd.to_numeric(non_blank.str.replace(",", "", regex=False), errors="coerce")
        if parsed.notna().mean() >= threshold:
            out[col] = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")
    return out
