#!/usr/bin/env python3
"""Acquire the predeclared 2026 Jan-Sep BTC/ETH holdout; integrity only.

No returns, market-direction summaries, features, fits, or strategy evaluations
are calculated. Original ZIPs and official SHA-256 CHECKSUMs are retained.
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
SYMBOLS = ("BTCUSDT", "ETHUSDT")
YEAR = 2026
MONTHS = tuple(range(1, 10))
BASE = "https://data.binance.vision/data/spot/monthly/klines"
README_URL = "https://raw.githubusercontent.com/binance/binance-public-data/master/README.md"
UTC = timezone.utc
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
FIELDS = ["open_time_utc", "open", "high", "low", "close", "volume", "close_time_utc",
          "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
          "taker_buy_quote_asset_volume"]


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def download(url):
    error = None
    for attempt in range(4):
        try:
            result = subprocess.run(
                ["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "45",
                 "--user-agent", "OxHack-holdout-integrity/1.0", url],
                check=True, capture_output=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as exc:
            error = exc.stderr.decode("utf-8", errors="replace").strip()
            if attempt < 3:
                time.sleep((attempt + 1) * .5)
    raise RuntimeError(f"Official archive unavailable: {url}: {error}")


def timestamp(value):
    # Official SPOT archive specification: Jan 2025 onward uses microseconds.
    seconds, microseconds = divmod(int(value), 1_000_000)
    return EPOCH + timedelta(seconds=seconds, microseconds=microseconds)


def iso(value):
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def month_data(symbol, month):
    name = f"{symbol}-1d-{YEAR}-{month:02d}.zip"
    url = f"{BASE}/{symbol}/1d/{name}"
    checksum_bytes = download(url + ".CHECKSUM")
    tokens = checksum_bytes.decode("utf-8").strip().split()
    if len(tokens) != 2 or tokens[1].lstrip("*") != name:
        raise ValueError(f"Unexpected official CHECKSUM format: {name}")
    expected_hash = tokens[0].lower()
    archive_path = RAW / symbol / name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    existing = archive_path.read_bytes() if archive_path.exists() else None
    content = existing if existing is not None and sha256(existing) == expected_hash else download(url)
    actual_hash = sha256(content)
    if actual_hash != expected_hash:
        raise ValueError(f"Official SHA-256 mismatch: {name}")
    archive_path.write_bytes(content)
    archive_path.with_suffix(".zip.CHECKSUM").write_bytes(checksum_bytes)
    member = name.removesuffix(".zip") + ".csv"
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if archive.namelist() != [member] or archive.testzip() is not None:
            raise ValueError(f"ZIP members/CRC failure: {name}")
        raw_csv = archive.read(member)
    original_rows = list(csv.reader(io.StringIO(raw_csv.decode("utf-8-sig"))))
    expected_count = calendar.monthrange(YEAR, month)[1]
    if len(original_rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows in {name}, received {len(original_rows)}")
    normalized = []
    for index, row in enumerate(original_rows):
        if len(row) != 12 or any(value == "" for value in row):
            raise ValueError(f"Missing/incorrect field count: {name} row {index + 1}")
        opened, closed = timestamp(row[0]), timestamp(row[6])
        if opened != datetime(YEAR, month, index + 1, tzinfo=UTC):
            raise ValueError(f"Daily UTC open/continuity failure: {name} row {index + 1}")
        if closed != opened + timedelta(days=1) - timedelta(microseconds=1):
            raise ValueError(f"Daily UTC close failure: {name} row {index + 1}")
        numeric = [Decimal(row[i]) for i in (1, 2, 3, 4, 5, 7, 9, 10, 11)]
        if not all(value.is_finite() for value in numeric):
            raise ValueError(f"Non-finite numeric field: {name}")
        op, high, low, close = numeric[:4]
        if not (0 < low <= op <= high and low <= close <= high):
            raise ValueError(f"OHLC relationship failure: {name}")
        if any(value < 0 for value in numeric[4:]) or int(row[8]) < 0:
            raise ValueError(f"Negative volume/trade count: {name}")
        if numeric[6] > numeric[4] or numeric[7] > numeric[5]:
            raise ValueError(f"Taker volume exceeds total: {name}")
        normalized.append([iso(opened), row[1], row[2], row[3], row[4], row[5], iso(closed),
                           row[7], row[8], row[9], row[10]])
    return normalized, {
        "symbol": symbol, "year": YEAR, "month": month, "zip_url": url,
        "checksum_url": url + ".CHECKSUM", "archive_path": str(archive_path.relative_to(ROOT)),
        "official_sha256": expected_hash, "actual_sha256": actual_hash,
        "checksum_verified": True, "zip_crc_verified": True,
        "raw_csv_sha256": sha256(raw_csv), "zip_bytes": len(content),
        "row_count": len(normalized), "timestamp_unit_in_source": "microseconds",
        "first_open_time_utc": normalized[0][0], "last_open_time_utc": normalized[-1][0],
    }


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    docs = DATA / "source_docs"
    docs.mkdir(exist_ok=True)
    documentation = download(README_URL)
    (docs / "binance-public-data-README.md").write_bytes(documentation)
    acquired = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(month_data, symbol, month): (symbol, month)
                   for symbol in SYMBOLS for month in MONTHS}
        for future in concurrent.futures.as_completed(futures):
            acquired[futures[future]] = future.result()
    annual_files = []
    start, end = datetime(YEAR, 1, 1, tzinfo=UTC), datetime(YEAR, 10, 1, tzinfo=UTC)
    expected_days = (end - start).days
    assert expected_days == 273
    for symbol in SYMBOLS:
        rows = [row for month in MONTHS for row in acquired[(symbol, month)][0]]
        dates = [row[0] for row in rows]
        if len(rows) != expected_days or len(set(dates)) != expected_days:
            raise ValueError(f"Combined period count/duplicate failure: {symbol}")
        if dates != [iso(start + timedelta(days=i)) for i in range(expected_days)]:
            raise ValueError(f"Combined period UTC continuity failure: {symbol}")
        target = DATA / f"{symbol}-1d-{YEAR}.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(FIELDS)
            writer.writerows(rows)
        annual_files.append({
            "symbol": symbol, "year": YEAR, "partial_calendar_year": True,
            "role": "primary_final_holdout" if symbol == "BTCUSDT" else "external_cross_asset_holdout_no_ETH_training",
            "path": str(target.relative_to(ROOT)), "sha256": sha256(target.read_bytes()),
            "row_count": len(rows), "start_inclusive_utc": iso(start), "end_exclusive_utc": iso(end),
            "missing_days": 0, "duplicate_open_times": 0,
            "ohlc_validation_passed": True, "volume_validation_passed": True,
            "daily_utc_continuity_passed": True,
        })
    manifest = {
        "dataset": "Binance SPOT BTCUSDT and ETHUSDT 1d klines; predeclared 2026 Jan-Sep holdouts",
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "official_documentation_url": "https://github.com/binance/binance-public-data",
        "official_kline_docs_url": "https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data",
        "data_root_url": BASE, "requires_api_key": False, "timezone": "UTC", "bar_interval": "1d",
        "predeclared_symbols": list(SYMBOLS), "year": YEAR, "months": list(MONTHS),
        "start_inclusive_utc": iso(start), "end_exclusive_utc": iso(end),
        "expected_days_per_symbol": expected_days,
        "timestamp_conversion": "All 2026 SPOT source timestamps are microseconds; integer quotient/remainder conversion to UTC ISO 8601, without floating-point timestamps.",
        "fields": FIELDS,
        "source_documents": [{"url": README_URL, "path": "data/source_docs/binance-public-data-README.md", "sha256": sha256(documentation)}],
        "monthly_archives": [acquired[(symbol, month)][1] for symbol in SYMBOLS for month in MONTHS],
        "annual_files": annual_files,
        "integrity": {"archive_count": 18, "all_official_checksums_verified": True,
                      "all_zip_crc_verified": True, "total_rows": expected_days * len(SYMBOLS),
                      "missing_days": 0, "duplicate_open_times_per_symbol": 0,
                      "validation_scope": "Data integrity only: hashes/CRC, exact dates, field presence, finite numeric values, OHLC relationships, volume consistency.",
                      "returns_or_strategy_results_computed": False},
        "separation": {"old_experiment_files_modified": False,
                       "old_2025_results_are_development_information": True,
                       "ETH_used_for_training": False,
                       "future_lock_required_before_performance_evaluation": True},
        "limitations": ["Single-exchange spot prices in USDT; not a consolidated global BTC/ETH market.",
                        "Daily OHLCV does not contain bid/ask spreads, order-book depth, intra-day trade ordering, or executable fills.",
                        "The final period/assets are fixed before evaluation; raw data acquisition/integrity checking is separate from model or strategy performance inspection.",
                        "Official archives can be revised; originals, official CHECKSUMs and local hashes are retained."],
    }
    (ROOT / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"availability": "all 18 monthly archives acquired and verified",
                      "rows_per_symbol": {r["symbol"]: r["row_count"] for r in annual_files},
                      "sha256_per_symbol": {r["symbol"]: r["sha256"] for r in annual_files},
                      "manifest": str(ROOT / "data_manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
