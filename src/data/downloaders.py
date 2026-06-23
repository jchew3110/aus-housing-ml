"""
Download ABS and RBA data files with URL discovery and local caching.

ABS download URLs are versioned per release (e.g. /dec-2024/641601.xlsx).
We scrape the /latest-release page each run to find the current URL, then
cache the file locally to avoid repeated downloads.
"""

import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.data.config import (
    ABS_CPI_FILE_PATTERN,
    ABS_CPI_RELEASE_URL,
    ABS_LF_FILE_PATTERN,
    ABS_LF_RELEASE_URL,
    ABS_RPPI_FILE_PATTERN,
    ABS_RPPI_RELEASE_URL,
    CACHE_TTL_SECONDS,
    DATA_RAW_DIR,
    RBA_CASH_RATE_URL,
)

logger = logging.getLogger(__name__)

ABS_BASE_URL = "https://www.abs.gov.au"


def discover_abs_download_url(release_page_url: str, file_pattern: str) -> str:
    """
    Scrape an ABS /latest-release page to find the direct file download URL.

    ABS HTML contains anchor tags like:
      <a href="/media/1234/641601.xlsx" class="...">Table 1. ...</a>

    Returns absolute URL to the matched file.
    Raises ValueError if no matching link is found.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    resp = requests.get(release_page_url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    pattern = re.compile(file_pattern, re.IGNORECASE)

    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if pattern.search(href):
            if href.startswith("http"):
                return href
            return ABS_BASE_URL + href

    raise ValueError(
        f"No link matching '{file_pattern}' found on {release_page_url}"
    )


def _is_cache_fresh(path: Path, ttl_seconds: int) -> bool:
    """Return True if path exists and was modified within ttl_seconds."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl_seconds


def download_file(
    url: str,
    dest_path: Path,
    cache_ttl_seconds: int = CACHE_TTL_SECONDS,
    force: bool = False,
) -> Path:
    """
    Download url to dest_path with caching.

    If dest_path exists and is fresh (within ttl), returns the cached path.
    Uses streaming download with a progress indicator.
    """
    if not force and _is_cache_fresh(dest_path, cache_ttl_seconds):
        logger.info("Cache hit: %s", dest_path.name)
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
    logger.info("Downloading %s -> %s", url, dest_path)

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

    logger.info("Saved %s (%.1f KB)", dest_path.name, dest_path.stat().st_size / 1024)
    return dest_path


def download_rppi(dest_dir: Path = DATA_RAW_DIR / "rppi", force: bool = False) -> Path:
    """Download ABS 6416.0 Table 1 (641601.xlsx) — RPPI by capital city."""
    url = discover_abs_download_url(ABS_RPPI_RELEASE_URL, ABS_RPPI_FILE_PATTERN)
    filename = url.split("/")[-1]
    return download_file(url, dest_dir / filename, force=force)


def download_cpi(dest_dir: Path = DATA_RAW_DIR / "cpi", force: bool = False) -> Path:
    """Download ABS 6401.0 Table 1 (640101.xlsx) — CPI all groups.

    Pinned to the Dec 2021 quarterly release to match the RPPI data range.
    The ABS switched to monthly CPI from Nov 2022; later releases of 640101.xlsx
    only contain recent monthly data rather than the full historical quarterly series.
    """
    url = (
        "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation"
        "/consumer-price-index-australia/dec-2021/640101.xlsx"
    )
    return download_file(url, dest_dir / "640101.xlsx", force=force)


def download_labour_force(
    dest_dir: Path = DATA_RAW_DIR / "labour_force", force: bool = False
) -> Path:
    """Download ABS 6202.0 Table 1 (62020001.xlsx) — Labour force summary."""
    url = discover_abs_download_url(ABS_LF_RELEASE_URL, ABS_LF_FILE_PATTERN)
    filename = url.split("/")[-1]
    return download_file(url, dest_dir / filename, force=force)


def download_rba_cash_rate(
    dest_dir: Path = DATA_RAW_DIR / "rba", force: bool = False
) -> Path:
    """Download RBA F1 table (f1-data.csv) — daily cash rate target."""
    return download_file(
        RBA_CASH_RATE_URL, dest_dir / "f1-data.csv", force=force
    )
