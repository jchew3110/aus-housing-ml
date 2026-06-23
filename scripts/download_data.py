#!/usr/bin/env python3
"""
CLI: download all data sources with caching.
Usage: python scripts/download_data.py [--force]
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.downloaders import (
    download_cpi,
    download_labour_force,
    download_rba_cash_rate,
    download_rppi,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Download ABS and RBA data files")
    parser.add_argument("--force", action="store_true", help="Ignore cache and re-download")
    args = parser.parse_args()

    print("Downloading RPPI (ABS 6416.0)...")
    path = download_rppi(force=args.force)
    print(f"  -> {path}")

    print("Downloading CPI (ABS 6401.0)...")
    path = download_cpi(force=args.force)
    print(f"  -> {path}")

    print("Downloading Labour Force (ABS 6202.0)...")
    path = download_labour_force(force=args.force)
    print(f"  -> {path}")

    print("Downloading RBA cash rate (F1)...")
    path = download_rba_cash_rate(force=args.force)
    print(f"  -> {path}")

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
