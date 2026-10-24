#!/usr/bin/env python3
"""Fetch official Binance BTCUSDT daily bars; validate integrity only.

This script deliberately computes no price returns, strategy results, or market
direction summaries. 2022-2025 are fixed calendar-year stages, chosen before
backtesting. All downloads are public and require no account/API key.
"""
from __future__ import annotations

import calendar
import concurrent.futures
import csv
import hashlib
import io
import json
import subprocess
import time
import zipfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
YEARS = (2022, 2023, 2024, 2025)
BASE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d"
PUBLIC_README = "https://raw.githubusercontent.com/binance/binance-public-data/master/README.md"
REST_DOCS = "https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/rest-api.md"
UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
FIELDS = [
    "open_time_utc", "open", "high", "low", "close", "volume", "close_time_utc",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    last_error = None
    for attempt in range(4):
        try:
            # macOS's bundled curl uses the system trust store. Keep HTTPS
            # verification enabled; some local Python runtimes lack CA bundles.
            response = subprocess.run(
                ["curl", "--fail", "--silent", "--show-error", "--location",
                 "--max-time", "45", "--user-agent", "OxHack-data-integrity/1.0", url],
                check=True, capture_output=True,
            )
            return response.stdout
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {last_error}")


def parse_timestamp(value: str, year: int) -> datetime:
    # Official SPOT archive specification changes on 2025-01-01.
    unit_divisor = 1_000_000 if year >= 2025 else 1_000
    raw = int(value)
    seconds, remainder = divmod(raw, unit_divisor)
    microseconds = remainder if year >= 2025 else remainder * 1_000
    return EPOCH + timedelta(seconds=seconds, microseconds=microseconds)


def iso(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def acquire_month(year: int, month: int) -> tuple[list[list[str]], dict]:
    name = f"BTCUSDT-1d-{year}-{month:02d}.zip"
    zip_url = f"{BASE}/{name}"
    checksum_url = zip_url + ".CHECKSUM"
    checksum_bytes = download(checksum_url)
    checksum_parts = checksum_bytes.decode("utf-8").strip().split()
    if len(checksum_parts) != 2 or checksum_parts[1].lstrip("*") != name:
        raise ValueError(f"Unexpected official CHECKSUM format for {name}")
    expected_hash = checksum_parts[0].lower()
    zip_path = RAW / name
    existing = zip_path.read_bytes() if zip_path.exists() else None
    zip_bytes = existing if existing is not None and sha256(existing) == expected_hash else download(zip_url)
    actual_hash = sha256(zip_bytes)
    if actual_hash != expected_hash:
        raise ValueError(f"Official SHA-256 verification FAILED for {name}")
    zip_path.write_bytes(zip_bytes)
    (RAW / f"{name}.CHECKSUM").write_bytes(checksum_bytes)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        members = archive.namelist()
        expected_csv = name.removesuffix(".zip") + ".csv"
        if members != [expected_csv]:
            raise ValueError(f"Unexpected ZIP entries in {name}: {members}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC failure for {name}: {bad_member}")
        raw_csv = archive.read(expected_csv)

    rows = list(csv.reader(io.StringIO(raw_csv.decode("utf-8-sig"))))
    expected_days = calendar.monthrange(year, month)[1]
    if len(rows) != expected_days:
        raise ValueError(f"Month {year}-{month:02d} has {len(rows)} rows, expected {expected_days}")
    normalized = []
    for index, row in enumerate(rows):
        if len(row) != 12 or any(value == "" for value in row):
            raise ValueError(f"Missing/incorrect fields in {name} row {index + 1}")
        opened = parse_timestamp(row[0], year)
        closed = parse_timestamp(row[6], year)
        expected_open = datetime(year, month, index + 1, tzinfo=UTC)
        tick = timedelta(microseconds=1 if year >= 2025 else 1_000)
        if opened != expected_open:
            raise ValueError(f"Daily UTC continuity/boundary failure in {name} row {index + 1}")
        if closed != opened + timedelta(days=1) - tick:
            raise ValueError(f"Daily close timestamp mismatch in {name} row {index + 1}")
        numeric = [Decimal(row[i]) for i in (1, 2, 3, 4, 5, 7, 9, 10, 11)]
        if not all(value.is_finite() for value in numeric):
            raise ValueError(f"Non-finite numeric field in {name}")
        op, high, low, close = numeric[:4]
        if not (0 < low <= op <= high and low <= close <= high):
            raise ValueError(f"Invalid OHLC relationship in {name}")
        if any(value < 0 for value in numeric[4:]) or int(row[8]) < 0:
            raise ValueError(f"Negative volume/count in {name}")
        if numeric[6] > numeric[4] or numeric[7] > numeric[5]:
            raise ValueError(f"Taker buy volume exceeds total volume in {name}")
        normalized.append([
            iso(opened), row[1], row[2], row[3], row[4], row[5], iso(closed),
            row[7], row[8], row[9], row[10],
        ])
    metadata = {
        "year": year, "month": month, "zip_url": zip_url,
        "checksum_url": checksum_url, "archive_path": str(zip_path.relative_to(ROOT)),
        "official_sha256": expected_hash, "actual_sha256": actual_hash,
        "checksum_verified": True, "zip_crc_verified": True,
        "raw_csv_sha256": sha256(raw_csv), "zip_bytes": len(zip_bytes),
        "row_count": len(normalized),
        "timestamp_unit_in_source": "microseconds" if year >= 2025 else "milliseconds",
        "first_open_time_utc": normalized[0][0], "last_open_time_utc": normalized[-1][0],
    }
    return normalized, metadata


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    docs = DATA / "source_docs"
    docs.mkdir(exist_ok=True)
    source_documents = []
    for name, url in (("binance-public-data-README.md", PUBLIC_README), ("binance-rest-api.md", REST_DOCS)):
        content = download(url)
        target = docs / name
        target.write_bytes(content)
        source_documents.append({"url": url, "path": str(target.relative_to(ROOT)), "sha256": sha256(content)})

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs = {pool.submit(acquire_month, year, month): (year, month)
                for year in YEARS for month in range(1, 13)}
        for future in concurrent.futures.as_completed(jobs):
            results[jobs[future]] = future.result()
    year_records = []
    all_times = []
    for stage, year in enumerate(YEARS, 1):
        rows = [row for month in range(1, 13) for row in results[(year, month)][0]]
        times = [row[0] for row in rows]
        expected_count = 366 if calendar.isleap(year) else 365
        if len(rows) != expected_count or len(set(times)) != expected_count:
            raise ValueError(f"Annual row count/duplicate failure for {year}")
        all_times.extend(times)
        output = DATA / f"BTCUSDT-1d-{year}.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            writer.writerows(rows)
        year_records.append({
            "stage": stage, "year": year, "path": str(output.relative_to(ROOT)),
            "sha256": sha256(output.read_bytes()), "row_count": len(rows),
            "start_inclusive_utc": f"{year}-01-01T00:00:00Z",
            "end_exclusive_utc": f"{year + 1}-01-01T00:00:00Z",
            "missing_days": 0, "duplicate_open_times": 0,
            "ohlc_validation_passed": True, "volume_validation_passed": True,
            "daily_utc_continuity_passed": True,
        })
    if len(all_times) != len(set(all_times)):
        raise ValueError("Duplicate times across annual stage boundaries")
    manifest = {
        "dataset": "Binance SPOT BTCUSDT 1d klines",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "official_documentation_url": "https://github.com/binance/binance-public-data",
        "official_kline_docs_url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data",
        "data_root_url": BASE, "requires_api_key": False,
        "calendar_years_fixed_before_backtesting": list(YEARS),
        "timezone": "UTC", "bar_interval": "1d", "symbol": "BTCUSDT",
        "base_asset": "BTC", "quote_asset": "USDT",
        "close_time_is_last_included_source_tick": True,
        "timestamp_conversion": "2022-2024 source milliseconds; 2025 source microseconds, per official SPOT archive specification. Convert using integer arithmetic to ISO 8601 UTC.",
        "fields": FIELDS, "source_documents": source_documents,
        "monthly_archives": [results[(year, month)][1] for year in YEARS for month in range(1, 13)],
        "annual_files": year_records,
        "integrity": {
            "archive_count": 48, "all_official_checksums_verified": True,
            "all_zip_crc_verified": True, "total_rows": len(all_times),
            "duplicate_open_times": 0, "missing_days": 0,
            "validation_scope": "Integrity, UTC boundaries/continuity, numeric finiteness, positive OHLC relationships, non-negative volumes/counts, taker-volume subset consistency only.",
            "returns_or_strategy_results_computed": False,
        },
        "limitations": [
            "These are Binance venue-specific BTC/USDT bars, not consolidated global BTC prices or a stock-market dataset.",
            "Daily OHLCV does not provide bid/ask spreads, order-book depth, trade ordering within a day, or executable fills.",
            "USDT is the quote asset; do not silently equate it with a guaranteed USD balance.",
            "Official archives can be revised; downloaded original ZIPs, official CHECKSUMs, source documentation snapshots and local SHA-256 hashes are retained.",
            "2025 is acquired and integrity-checked only; no market direction, return, or strategy-performance summary is produced by this script.",
        ],
    }
    (ROOT / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"annual_rows": {record["year"]: record["row_count"] for record in year_records},
                      "total_rows": len(all_times), "verified_archives": 48,
                      "manifest": str(ROOT / "data_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
