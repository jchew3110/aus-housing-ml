"""Tests for download caching logic."""

import time

from src.data.downloaders import _is_cache_fresh, download_file


class TestIsCacheFresh:
    def test_missing_file_is_not_fresh(self, tmp_path):
        assert not _is_cache_fresh(tmp_path / "nonexistent.xlsx", ttl_seconds=3600)

    def test_fresh_file_is_fresh(self, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"data")
        assert _is_cache_fresh(f, ttl_seconds=3600)

    def test_old_file_is_not_fresh(self, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"data")
        # Backdate mtime by 2 hours
        old_time = time.time() - 7200
        import os
        os.utime(f, (old_time, old_time))
        assert not _is_cache_fresh(f, ttl_seconds=3600)


class TestDownloadFile:
    def test_cached_file_not_re_downloaded(self, tmp_path, requests_mock):
        dest = tmp_path / "test.csv"
        dest.write_bytes(b"cached")
        url = "http://example.com/test.csv"
        requests_mock.get(url, text="new content")

        result = download_file(url, dest, cache_ttl_seconds=3600)
        assert result == dest
        # File should still contain cached content
        assert dest.read_bytes() == b"cached"
        assert not requests_mock.called

    def test_downloads_when_cache_missing(self, tmp_path, requests_mock):
        dest = tmp_path / "test.csv"
        url = "http://example.com/test.csv"
        requests_mock.get(url, content=b"downloaded content")

        result = download_file(url, dest, cache_ttl_seconds=3600)
        assert result == dest
        assert dest.read_bytes() == b"downloaded content"

    def test_force_redownloads(self, tmp_path, requests_mock):
        dest = tmp_path / "test.csv"
        dest.write_bytes(b"old")
        url = "http://example.com/test.csv"
        requests_mock.get(url, content=b"new")

        download_file(url, dest, cache_ttl_seconds=3600, force=True)
        assert dest.read_bytes() == b"new"
