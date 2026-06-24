"""
Orchestrates the full data ingestion pipeline:
  download → parse → resample → merge → save parquet
"""

import logging

import pandas as pd

from src.data.config import DATA_PROCESSED_DIR
from src.data.downloaders import (
    download_cpi,
    download_labour_force,
    download_rba_cash_rate,
    download_rppi,
)
from src.data.parsers import (
    merge_to_panel,
    parse_cpi,
    parse_labour_force,
    parse_rba_csv,
    parse_rppi,
    resample_to_quarter,
)

logger = logging.getLogger(__name__)

PANEL_PATH = DATA_PROCESSED_DIR / "features_panel.parquet"


def run_data_pipeline(force_download: bool = False) -> pd.DataFrame:
    """
    Full ingestion pipeline. Downloads data if not cached, parses all sources,
    resamples to quarterly, merges to a long-format panel, and saves parquet.

    Returns the merged panel DataFrame.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --- Download ---
    logger.info("Downloading data sources...")
    rppi_path = download_rppi(force=force_download)
    cpi_path = download_cpi(force=force_download)
    lf_path = download_labour_force(force=force_download)
    rba_path = download_rba_cash_rate(force=force_download)

    # --- Parse ---
    logger.info("Parsing ABS RPPI...")
    rppi_df = parse_rppi(rppi_path)

    logger.info("Parsing ABS CPI...")
    cpi_series = parse_cpi(cpi_path)

    logger.info("Parsing ABS Labour Force...")
    lf_series = parse_labour_force(lf_path)

    logger.info("Parsing RBA cash rate...")
    rba_series = parse_rba_csv(rba_path)

    # --- Resample macro series to quarterly ---
    cash_rate_q = resample_to_quarter(rba_series, method="last")
    lf_q = resample_to_quarter(lf_series, method="last")
    # CPI may already be quarterly; resample is a no-op if already Q freq
    if not isinstance(cpi_series.index, pd.PeriodIndex):
        cpi_q = resample_to_quarter(cpi_series, method="last")
    else:
        cpi_q = cpi_series

    # --- Merge ---
    logger.info("Merging to panel...")
    panel = merge_to_panel(rppi_df, cash_rate_q, cpi_q, lf_q)

    logger.info("Panel shape: %s", panel.shape)
    logger.info("Cities: %s", sorted(panel["city"].unique()))
    logger.info("Period range: %s — %s", panel["period"].min(), panel["period"].max())

    # --- Save ---
    panel.to_parquet(PANEL_PATH, index=False)
    logger.info("Saved panel to %s", PANEL_PATH)

    return panel


def load_panel() -> pd.DataFrame:
    """Load the processed panel from parquet. Run run_data_pipeline() first."""
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"{PANEL_PATH} not found. Run `make process-data` first."
        )
    df = pd.read_parquet(PANEL_PATH)
    # Restore Period dtype
    df["period"] = pd.PeriodIndex(df["period"], freq="Q-DEC")
    return df
