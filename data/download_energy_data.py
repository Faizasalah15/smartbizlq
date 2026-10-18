"""
Download script for Northern Ireland electricity demand + Belfast weather data.

Sources:
    1. EirGrid / SONI Smart Grid Dashboard — NI system demand (15-min resolution)
       URL: https://www.smartgriddashboard.com/
       Licence: Proprietary © EirGrid plc & SONI Ltd. Small samples permitted with attribution.

    2. Open-Meteo Historical Weather API — Belfast temperature (hourly)
       URL: https://open-meteo.com/
       Licence: CC-BY 4.0 (Copernicus ERA5 reanalysis data)

Usage:
    python data/download_energy_data.py --start 2023-01-01 --end 2023-12-31
    python data/download_energy_data.py --demand-csv my_download.csv   # manual fallback

Note: the EirGrid endpoint is the one the dashboard uses internally, not a
documented public API, so it may change. The manual CSV route always works.

The script merges both sources into a single hourly CSV suitable for
demand forecasting with temperature as an external regressor.
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── EirGrid / SONI Smart Grid Dashboard ────────────────────────────────
# Unofficial JSON endpoint used by the dashboard itself. It is not a documented
# public API and may change; if it fails, use --demand-csv with a file downloaded
# by hand from https://www.smartgriddashboard.com (Northern Ireland > Demand > Download CSV).
EIRGRID_DATA_URL = "https://www.smartgriddashboard.com/DashboardService.svc/data"


def _parse_demand_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Find the timestamp and demand columns in a dashboard export and standardise them."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c: c.lower() for c in df.columns}
    dt_col = next((c for c in df.columns if any(k in lower[c] for k in ("effectivetime", "datetime", "date", "time"))), None)
    val_col = next((c for c in df.columns if "demand" in lower[c] and c != dt_col), None)
    if val_col is None:
        val_col = next((c for c in df.columns if lower[c] == "value"), None)
    if dt_col is None or val_col is None:
        raise ValueError(f"Could not find timestamp/demand columns in {list(df.columns)}")
    out = pd.DataFrame({
        "datetime": pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True),
        "demand_mw": pd.to_numeric(df[val_col], errors="coerce"),
    })
    return out.dropna()


def download_ni_demand(start: str, end: str) -> pd.DataFrame:
    """Download Northern Ireland actual system demand (MW, 15-minute) from EirGrid.

    Requests are made one month at a time. Returns an empty frame if the
    endpoint is unavailable, so the caller can fall back to --demand-csv.
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    chunks: list[pd.DataFrame] = []
    current = start_dt
    while current <= end_dt:
        chunk_end = min(current + timedelta(days=30), end_dt)
        params = {
            "area": "demandactual",
            "region": "NI",
            "datefrom": current.strftime("%d-%b-%Y") + " 00:00",
            "dateto": chunk_end.strftime("%d-%b-%Y") + " 23:59",
        }
        logger.info("Fetching NI demand %s to %s", current.date(), chunk_end.date())
        try:
            resp = requests.get(EIRGRID_DATA_URL, params=params, timeout=60)
            resp.raise_for_status()
            rows = resp.json().get("Rows", [])
            if rows:
                chunks.append(_parse_demand_frame(pd.DataFrame(rows)))
            else:
                logger.warning("No rows returned for this period")
        except Exception as exc:
            logger.warning("Demand request failed: %s", exc)
            if not chunks:
                break  # endpoint unavailable; stop early instead of retrying every month
        current = chunk_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame(columns=["datetime", "demand_mw"])
    demand = pd.concat(chunks, ignore_index=True).sort_values("datetime").drop_duplicates("datetime")
    logger.info("Downloaded %d demand records", len(demand))
    return demand


def load_demand_csv(path: str) -> pd.DataFrame:
    """Load a demand CSV downloaded manually from the Smart Grid Dashboard."""
    demand = _parse_demand_frame(pd.read_csv(path))
    logger.info("Loaded %d demand records from %s", len(demand), path)
    return demand.sort_values("datetime").drop_duplicates("datetime")


# ── Open-Meteo Historical Weather API ──────────────────────────────────
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Belfast, Northern Ireland coordinates
BELFAST_LAT = 54.5973
BELFAST_LON = -5.9301


def download_belfast_weather(start: str, end: str) -> pd.DataFrame:
    """Download hourly Belfast temperature from Open-Meteo (ERA5 reanalysis).

    Args:
        start: Start date in YYYY-MM-DD format.
        end: End date in YYYY-MM-DD format.

    Returns:
        DataFrame with columns [datetime, temperature_c, wind_speed_kmh].
    """
    params = {
        "latitude": BELFAST_LAT,
        "longitude": BELFAST_LON,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,wind_speed_10m",
        "timezone": "Europe/London",
    }
    logger.info("Fetching Belfast weather: %s to %s", start, end)
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        weather_df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(hourly.get("time", []), errors="coerce"),
                "temperature_c": hourly.get("temperature_2m", []),
                "wind_speed_kmh": hourly.get("wind_speed_10m", []),
            }
        )
        weather_df = weather_df.dropna(subset=["datetime"])
        logger.info("Downloaded %d weather records", len(weather_df))
        return weather_df

    except Exception as exc:
        logger.error("Failed to fetch weather data: %s", exc)
        return pd.DataFrame(columns=["datetime", "temperature_c", "wind_speed_kmh"])


# ── Merge & Export ─────────────────────────────────────────────────────

def merge_demand_weather(
    demand_df: pd.DataFrame, weather_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge demand and weather data on hourly timestamps.

    Demand data (15-min) is resampled to hourly means.
    Weather data is already hourly.

    Returns:
        Merged DataFrame with columns:
        [datetime, demand_mw, temperature_c, wind_speed_kmh, hour, day_of_week, month]
    """
    # Resample demand to hourly
    demand_hourly = (
        demand_df
        .set_index("datetime")
        .resample("h")
        .mean()
        .reset_index()
    )

    # Round weather timestamps to hour
    weather_df = weather_df.copy()
    weather_df["datetime"] = weather_df["datetime"].dt.floor("h")

    # Merge on datetime
    merged = pd.merge(demand_hourly, weather_df, on="datetime", how="inner")

    # Add time features useful for forecasting
    merged["hour"] = merged["datetime"].dt.hour
    merged["day_of_week"] = merged["datetime"].dt.dayofweek
    merged["month"] = merged["datetime"].dt.month

    logger.info("Merged dataset: %d rows", len(merged))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NI electricity demand + Belfast weather data"
    )
    parser.add_argument(
        "--start", default="2023-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default="2023-12-31", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "sample", "ni_energy_demand.csv"
        ),
        help="Output CSV path",
    )
    parser.add_argument(
        "--demand-csv",
        help="Use a demand CSV downloaded by hand from smartgriddashboard.com instead of downloading",
    )
    args = parser.parse_args()

    demand = load_demand_csv(args.demand_csv) if args.demand_csv else download_ni_demand(args.start, args.end)
    if args.demand_csv and not demand.empty:
        # Fetch weather for exactly the period covered by the manual file
        args.start = demand["datetime"].min().strftime("%Y-%m-%d")
        args.end = demand["datetime"].max().strftime("%Y-%m-%d")
    weather = download_belfast_weather(args.start, args.end)

    if demand.empty or weather.empty:
        logger.error(
            "One or both data sources returned no data. If the EirGrid download failed, "
            "download NI demand as CSV from https://www.smartgriddashboard.com and run: "
            "python data/download_energy_data.py --demand-csv <file.csv>"
        )
        sys.exit(1)

    merged = merge_demand_weather(demand, weather)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.to_csv(args.output, index=False)
    logger.info("Saved merged dataset to %s (%d rows)", args.output, len(merged))


if __name__ == "__main__":
    main()
