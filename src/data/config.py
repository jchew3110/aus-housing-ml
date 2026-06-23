from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

CITIES = [
    "Sydney",
    "Melbourne",
    "Brisbane",
    "Adelaide",
    "Perth",
    "Hobart",
    "Darwin",
    "Canberra",
]

# ABS release page URLs — scrape these to discover the versioned download link
ABS_RPPI_RELEASE_URL = (
    "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation"
    "/residential-property-price-indexes-eight-capital-cities/latest-release"
)
ABS_CPI_RELEASE_URL = (
    "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation"
    "/consumer-price-index-australia/latest-release"
)
ABS_LF_RELEASE_URL = (
    "https://www.abs.gov.au/statistics/labour/employment-and-unemployment"
    "/labour-force-australia/latest-release"
)
RBA_CASH_RATE_URL = "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv"

# File patterns to match on ABS release pages
ABS_RPPI_FILE_PATTERN = r"641601\.xlsx"
ABS_CPI_FILE_PATTERN = r"640101\.xlsx"
ABS_LF_FILE_PATTERN = r"620[12]0001\.xlsx"

# Cache TTL: re-download files older than this (seconds)
CACHE_TTL_SECONDS = 7 * 24 * 3600  # 1 week


@dataclass
class SplitConfig:
    train_end: str = "2018Q4"
    val_end: str = "2020Q4"
    # test = everything after val_end (2021Q1–2021Q4 given current RPPI range)
