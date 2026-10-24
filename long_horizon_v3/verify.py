"""Independent, scalar accounting audit of the long-horizon historical replays.

No production planner, replay, financial-control, EMA, or summary helper is
imported.  This checks the recorded orders against real OHLC prices, independently
executes all transactions, and reconciles every account with episodes.csv.
It does not turn the reused 2026 evaluation data into a new unseen holdout.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "time_budget_policy_v2"
BF, SF = 1.0005 * 1.001, .9995 * .999
CONTROLLERS = {"dp30", "dp2", "dp30_no_mid_buy", "ema_time", "ema_mechanical", "retire_now"}
EXPECTED_STARTS = list(range(30, 243))
COUNTS = defaultdict(int)
BOOTSTRAP_WEIGHTS = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b, label):
    COUNTS["scalar_numeric_checks"] += 1
    if not math.isclose(float(a), float(b), abs_tol=3e-8, rel_tol=3e-10):
        raise AssertionError(f"{label}: {a!r} != {b!r}")


def flag(value):
    return str(value).lower() in {"true", "1"}


def u(cash, life, alpha):
    if life <= 0:
        return 0.
    return alpha * math.log1p(max(0., life) / 90.) + (1 - alpha) * math.log1p(max(0., cash) / 100.)


def settle(cash, holding, life, packs, alpha, price):
    """Independent exhaustive terminal pack choice, smallest-count ties."""
    base_cash = cash + holding * SF
    if life <= 0:
        return dict(cash=0., holding=0., life=0., packs=packs, utility=0.,
                    commission=0., slippage_cost=0., fees=0., turnover=0., trades=0,
                    life_spend=0., packs_bought=0, mid_packs=0, terminal_packs=0)
    best, count = u(base_cash, life, alpha), 0
    for j in range(1, 7 - packs):
        if base_cash + 1e-9 >= j * price:
            candidate = u(max(0., base_cash - j * price), life + j * 15, alpha)
            if candidate > best + 1e-12:
                best, count = candidate, j
    commission, slip = holding * .9995 * .001, holding * .0005
    return dict(cash=max(0., base_cash - count * price), holding=0.,
                life=life + count * 15, packs=packs + count, utility=best,
                commission=commission, slippage_cost=slip, fees=commission + slip,
                turnover=holding, trades=int(holding > 1e-10), life_spend=count * price,
                packs_bought=count, mid_packs=0, terminal_packs=count)


def controls(cash, holding, life, packs, action, price):
    """Execute one order at a known open; assets are marked monetary values."""
    target = (0., 0., .5, 1., 0., .5, 1.)[action]
    buying_time = action >= 4
    if life <= 0 or (buying_time and (packs >= 6 or cash + holding * SF + 1e-9 < price)):
        return None
    time_sale = 0.
    if buying_time:
        time_sale = min(max(0., price - cash) / SF, holding)
        cash = max(0., cash + time_sale * SF - price)
        holding -= time_sale
        life += 15
        packs += 1
    difference = target * (cash + holding) - holding
    buy = sell = 0.
    if difference > 0:
        buy = difference / (1 + target * (BF - 1))
        cash -= buy * BF
        holding += buy
    elif difference < 0:
        sell = -difference / (1 - target * (1 - SF))
        cash += sell * SF
        holding -= sell
    commission = ((time_sale + sell) * .9995 + buy * 1.0005) * .001
    slip = (time_sale + sell + buy) * .0005
    return dict(cash=max(0., cash), holding=max(0., holding), life=life, packs=packs,
                commission=commission, slippage_cost=slip, fees=commission + slip,
                turnover=time_sale + sell + buy,
                trades=int(time_sale > 1e-10) + int(buy > 1e-10) + int(sell > 1e-10),
                life_spend=price if buying_time else 0., packs_bought=int(buying_time),
                mid_packs=int(buying_time), terminal_packs=0)


def one_event(cash, holding, life, packs, action, gap, body, remaining, alpha, price):
    if action == 0:
        result = settle(cash, holding * gap, life, packs, alpha, price)
        result.update(done=True, reason="retire", consumed=0)
        return result
    result = controls(cash, holding * gap, life, packs, action, price)
    if result is None:
        return None
    result["holding"] *= body
    result["life"] -= 5
    result.update(done=False, reason="continue", utility=0., consumed=1)
    if result["life"] <= 0:
        result.update(cash=0., holding=0., life=0., done=True, reason="death", utility=0.)
    elif remaining == 1:
        terminal = settle(result["cash"], result["holding"], result["life"], result["packs"], alpha, price)
        for field in ("cash", "holding", "life", "packs", "utility"):
            result[field] = terminal[field]
        for field in ("commission", "slippage_cost", "fees", "turnover", "trades", "life_spend", "packs_bought", "terminal_packs"):
            result[field] += terminal[field]
        result.update(done=True, reason="horizon")
    return result


def account_key(row):
    return (row["symbol"], row["controller"], float(row["pack_cost"]), float(row["alpha"]),
            float(row["initial_life"]), int(float(row["start_index"])))


def load_market(path, volatility_threshold):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    opens, closes = ([float(r[field]) for r in rows] for field in ("open", "close"))
    dates = [r["open_time_utc"] for r in rows]
    parsed = [datetime.fromisoformat(s.replace("Z", "+00:00")) for s in dates]
    assert len(rows) == 273 and parsed[0].date().isoformat() == "2026-01-01"
    assert parsed[-1].date().isoformat() == "2026-09-30"
    assert all(b - a == timedelta(days=1) for a, b in zip(parsed, parsed[1:]))
    states = {}
    for t in range(21, len(rows)):
        momentum = closes[t - 1] / closes[t - 6] - 1
        vol = np.std([math.log(closes[j] / closes[j - 1]) for j in range(t - 20, t)], ddof=1)
        states[t] = 2 * int(momentum >= 0) + int(vol >= volatility_threshold)
    def ema(period):
        values = [None] * len(closes)
        value = sum(closes[:period]) / period
        values[period - 1] = value
        coefficient = 2. / (period + 1)
        for i in range(period, len(closes)):
            value = coefficient * closes[i] + (1 - coefficient) * value
            values[i] = value
        return values
    return dict(rows=rows, opens=opens, closes=closes, dates=dates, states=states,
                fast=ema(15), slow=ema(30))


def audit_setting(directory, market, *, allow_partial=False):
    with (directory / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    episode_lookup = {account_key(row): row for row in episodes}
    assert len(episode_lookup) == len(episodes), "duplicate account"
    accounts = {}
    totals = ("commission", "slippage_cost", "fees", "turnover", "trades", "life_spend", "packs_bought", "mid_packs", "terminal_packs")
    ledger_events = 0
    with gzip.open(directory / "ledger.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = account_key(row)
            symbol, controller, price, alpha, initial_life, start = key
            assert controller in CONTROLLERS and price in (0., 2., 8.)
            assert key in episode_lookup
            if key not in accounts:
                accounts[key] = dict(cash=100., holding=0., life=initial_life, packs=0,
                    step=0, consumed=0, done=False, totals={field: 0. for field in totals})
            account = accounts[key]
            k = int(row["step"])
            assert not account["done"] and k == account["step"] and 0 <= k < 30
            data = market[symbol]
            day = start + k
            assert row["date"] == data["dates"][day]
            assert int(row["market_state"]) == data["states"][day]
            for field in ("cash", "holding", "life", "packs"):
                same(account[field], row[field + "_before"], "ledger state continuity")
            gap = data["opens"][day] / data["closes"][day - 1]
            body = data["closes"][day] / data["opens"][day]
            same(gap, row["gap"], "real opening gap")
            same(body, row["intraday"], "real intraday return")
            intended, executed = int(row["intended_action"]), int(row["executed_action"])
            assert 0 <= intended <= 6 and 0 <= executed <= 6
            if controller == "retire_now":
                assert k == 0 and intended == 0
            if controller == "dp30_no_mid_buy":
                assert intended < 4
            if controller in {"ema_time", "ema_mechanical"}:
                fast, slow = data["fast"][day - 1], data["slow"][day - 1]
                bullish = fast > slow * 1.00015
                invested = fast >= slow if account["holding"] > 1e-10 else bullish
                expected_action = 3 if invested else 1
                time_request = (controller == "ema_mechanical" and account["life"] <= 10
                                and account["packs"] < 6 and bullish)
                if time_request:
                    expected_action += 3
                elif account["life"] <= 5:
                    expected_action = 0
                assert intended == expected_action, "EMA order must use only previous-close indicators"
            result = one_event(account["cash"], account["holding"], account["life"], account["packs"],
                               intended, gap, body, 30 - k, alpha, price)
            if result is None:
                protected_action = 0 if account["life"] <= 5 else intended - 3
                assert executed == protected_action, "wrong order rejection protection"
                result = one_event(account["cash"], account["holding"], account["life"], account["packs"],
                                   executed, gap, body, 30 - k, alpha, price)
            else:
                assert executed == intended
            assert result is not None
            for field in ("cash", "holding", "life", "packs"):
                same(result[field], row[field + "_after"], "independently executed state")
            for field in totals:
                same(result[field], row[field], "independently executed costs/packs")
                account["totals"][field] += result[field]
            same(result["utility"], row["actual_utility"], "event terminal utility")
            assert result["done"] == flag(row["done"])
            assert result["reason"] == row["terminal_reason"]
            account.update({field: result[field] for field in ("cash", "holding", "life", "packs")})
            account["done"] = result["done"]
            account["reason"] = result["reason"]
            account["step"] += 1
            account["consumed"] += result["consumed"]
            account["last_date"] = row["date"]
            ledger_events += 1
    assert set(accounts) == set(episode_lookup), "ledger/episode account coverage"
    groups = defaultdict(list)
    for key, account in accounts.items():
        assert account["done"], "unfinished account"
        symbol, controller, price, alpha, initial_life, start = key
        row = episode_lookup[key]
        for field, actual in (("final_cash", account["cash"]), ("remaining_life", account["life"]),
                              ("utility", u(account["cash"], account["life"], alpha)),
                              ("total_life_spend", account["totals"]["life_spend"]),
                              ("mid_packs", account["totals"]["mid_packs"]),
                              ("terminal_packs", account["totals"]["terminal_packs"]),
                              ("fees", account["totals"]["fees"]),
                              ("trades", account["totals"]["trades"]),
                              ("exit_step", account["consumed"]),
                              ("trading_pnl", account["cash"] + account["totals"]["life_spend"] - 100.)):
            same(actual, row[field], "account ledger/episode reconciliation")
        assert account["reason"] == row["reason"]
        assert row["start_date"] == market[symbol]["dates"][start]
        assert row["exit_date"] == account["last_date"]
        same(account["totals"]["packs_bought"], account["packs"], "pack count conservation")
        same(account["totals"]["life_spend"], price * account["packs"], "time spend conservation")
        same(account["totals"]["fees"], account["totals"]["commission"] + account["totals"]["slippage_cost"], "fee parts")
        if account["life"] > 0:
            same(account["life"], initial_life + 15 * account["packs"] - 5 * account["consumed"], "life conservation")
        groups[(symbol, controller, price, alpha, initial_life)].append(row)
    aggregates = []
    for key, rows in sorted(groups.items()):
        starts = sorted(int(row["start_index"]) for row in rows)
        if not allow_partial:
            assert starts == EXPECTED_STARTS, "all controllers need the same complete starts"
        aggregates.append(dict(zip(("symbol", "controller", "pack_cost", "alpha", "initial_life"), key)) | {
            "n": len(rows), "mean_utility": float(np.mean([float(r["utility"]) for r in rows])),
            "mean_final_cash": float(np.mean([float(r["final_cash"]) for r in rows])),
            "mean_remaining_life": float(np.mean([float(r["remaining_life"]) for r in rows])),
            "mean_mid_packs": float(np.mean([float(r["mid_packs"]) for r in rows])),
            "mean_terminal_packs": float(np.mean([float(r["terminal_packs"]) for r in rows])),
            "mean_exit_step": float(np.mean([float(r["exit_step"]) for r in rows])),
        })
    if not allow_partial:
        assert {key[0] for key in groups} == {"BTCUSDT", "ETHUSDT"}
        assert {key[1] for key in groups} == CONTROLLERS
        assert len(episodes) == 2 * len(CONTROLLERS) * len(EXPECTED_STARTS)
    return {"setting": directory.name, "accounts": len(accounts), "ledger_events": ledger_events,
            "episodes_sha256": digest(directory / "episodes.csv"),
            "ledger_sha256": digest(directory / "ledger.csv.gz"), "independent_aggregates": aggregates}


def bootstrap_interval(values, block):
    """Independent resampling by frequency counts, rather than gathered arrays."""
    values = np.asarray(values, float)
    n = len(values)
    key = (n, block)
    if key not in BOOTSTRAP_WEIGHTS:
        rng = np.random.default_rng(20261024)
        weights = np.zeros((2000, n), float)
        for i in range(2000):
            origins = rng.integers(0, n, size=math.ceil(n / block))
            sampled = []
            for first in origins:
                sampled.extend((int(first) + j) % n for j in range(block))
            weights[i] = np.bincount(sampled[:n], minlength=n) / n
        BOOTSTRAP_WEIGHTS[key] = weights
    return np.percentile(BOOTSTRAP_WEIGHTS[key] @ values, [2.5, 97.5])


def original_ema_wealth(data, start):
    """Literal independent EMA account: no life clock or time-pack conversion."""
    cash, holding = 100., 0.
    for day in range(start, start + 30):
        fast, slow = data["fast"][day - 1], data["slow"][day - 1]
        full = fast >= slow if holding > 1e-10 else fast > slow * 1.00015
        gap = data["opens"][day] / data["closes"][day - 1]
        body = data["closes"][day] / data["opens"][day]
        out = controls(cash, holding * gap, 90., 0, 3 if full else 1, 8.)
        cash, holding = out["cash"], out["holding"] * body
    return cash + holding * SF


def audit_summary(directories, market):
    path = ROOT / "results.json"
    assert path.exists(), "full verification requires final results.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    assert result["source_hashes_after"] == protocol["source_hashes"]
    groups = defaultdict(list)
    n_episodes = 0
    for directory in directories:
        with (directory / "episodes.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                symbol, controller, price, alpha, life, start = account_key(row)
                groups[(symbol, price, alpha, life, controller)].append(row)
                n_episodes += 1
    assert result["episodes"] == n_episodes
    account_rows = {account_key(row): row for rows in groups.values() for row in rows}
    combined_seen = set()
    with (ROOT / "episodes.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = account_key(row)
            assert key in account_rows and key not in combined_seen
            combined_seen.add(key)
            original_row = account_rows[key]
            for field in ("final_cash", "remaining_life", "utility", "total_life_spend", "mid_packs", "terminal_packs", "fees", "trades", "exit_step", "trading_pnl"):
                same(row[field], original_row[field], "downloadable combined CSV matches per-setting source")
            assert row["reason"] == original_row["reason"]
    assert combined_seen == set(account_rows)
    summaries = {(r["symbol"], float(r["pack_cost"]), float(r["alpha"]), float(r["initial_life"]), r["controller"]): r
                 for r in result["comparison"]}
    assert set(summaries) == set(groups)
    metrics = ("final_cash", "remaining_life", "utility", "total_life_spend", "mid_packs", "terminal_packs", "fees", "trades", "exit_step", "trading_pnl")
    for key, rows in groups.items():
        rows.sort(key=lambda row: int(row["start_index"]))
        target = summaries[key]
        assert target["n"] == len(rows) == 213
        for metric in metrics:
            same(np.mean([float(r[metric]) for r in rows]), target[metric], "summary mean")
        for field, criterion in (("immediate_exit_rate", lambda r: int(r["exit_step"]) == 0),
                                 ("death_rate", lambda r: r["reason"] == "death"),
                                 ("mid_buy_rate", lambda r: int(r["mid_packs"]) > 0),
                                 ("terminal_buy_rate", lambda r: int(r["terminal_packs"]) > 0)):
            same(np.mean([criterion(r) for r in rows]), target[field], "summary rate")
    contrasts = result["contrasts"]
    assert len(contrasts) == len(groups) // 6 * 5
    for row in contrasts:
        identity = (row["symbol"], float(row["pack_cost"]), float(row["alpha"]), float(row["initial_life"]))
        left, right = groups[identity + ("dp30",)], groups[identity + (row["baseline"],)]
        assert [r["start_index"] for r in left] == [r["start_index"] for r in right]
        d = np.array([float(a["utility"]) - float(b["utility"]) for a, b in zip(left, right)])
        same(d.mean(), row["effect"], "paired mean")
        same(d[::31].mean(), row["nonoverlap_effect"], "nonoverlap paired mean")
        assert len(d[::31]) == row["nonoverlap_n"] == 7
        same(np.mean([float(a["final_cash"]) - float(b["final_cash"]) for a, b in zip(left, right)]), row["cash_effect"], "paired cash effect")
        for block in (30, 60):
            lo, hi = bootstrap_interval(d, block)
            same(lo, row[f"low{block}"], "independent circular block bootstrap lower")
            same(hi, row[f"high{block}"], "independent circular block bootstrap upper")
    for row in result["original_ema"]:
        assert row["n"] == 213 and "utility" not in row
        values = [original_ema_wealth(market[row["symbol"]], start) for start in EXPECTED_STARTS]
        same(np.mean(values), row["final_cash"], "independent original no-life EMA wealth")
    diagnostics_checked = 0
    for setting in result["diagnostics"]:
        for name in ("full", "no_mid_buy"):
            meta = setting[name]
            assert meta["horizon_steps"] == 30 and meta["dynamic_bounds"]
            same(meta["maximum_internal_wealth_extrapolation"], 0., "internal grid closure")
            q = meta["query_diagnostics"]
            assert q["queries"] > 0, "diagnostics must be captured after the actual replay"
            assert q["root_children_above_grid"] == q["root_children_below_grid"] == q["future_reachability_failures"] == 0
            diagnostics_checked += 1
    return {"summary_groups": len(groups), "paired_contrasts": len(contrasts),
            "bootstrap_intervals": len(contrasts) * 2, "original_ema_assets": len(result["original_ema"]),
            "replay_domain_diagnostics": diagnostics_checked, "results_sha256": digest(path)}


def audit_purchase_cases():
    """Light addendum: check a post-hoc illustration against frozen ledgers."""
    path = ROOT / "purchase_cases.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["status"] == "complete" and value["settings_complete"] == 24
    captured = {}
    source_counts = {}
    for ledger in sorted((ROOT / "runs").glob("*/ledger.csv.gz")):
        assert (ledger.parent / "complete.json").exists()
        count = 0
        with gzip.open(ledger, "rt", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["controller"] != "dp30" or int(float(row["mid_packs"])) <= 0:
                    continue
                identity = account_key(row) + (int(row["step"]), row["date"])
                assert identity not in captured
                captured[identity] = row
                count += 1
        source_counts[str(ledger.resolve())] = count
    assert len(source_counts) == len(value["sources"]) == 24
    for source in value["sources"]:
        ledger = Path(source["ledger"])
        assert str(ledger.resolve()) in source_counts
        assert source["ledger_sha256"] == digest(ledger)
        assert source["mid_purchase_events"] == source_counts[str(ledger.resolve())]
    seen = set()
    def sort_key(event):
        return (event["date"], event["symbol"], int(event["start_index"]), int(event["step"]),
                float(event["pack_cost"]), float(event["alpha"]), float(event["initial_life"]))
    events = value["events"]
    assert len(events) == len(captured) == value["event_count"]
    for event in events:
        identity = account_key(event) + (int(event["step"]), event["date"])
        assert identity in captured and identity not in seen
        seen.add(identity)
        row = captured[identity]
        for field in ("cash_before", "holding_before", "life_before", "packs_before", "market_state", "intended_action", "executed_action", "mid_packs"):
            same(event[field], row[field], "purchase illustration exact ledger state")
        assert event["steps_remaining"] == 30 - int(row["step"])
        assert event["executed_action"] >= 4
        assert event["executed_action_name"] == ("retire", "cash", "half", "full", "buy_time_cash", "buy_time_half", "buy_time_full")[event["executed_action"]]
        assert event["price_label"] == ("free" if event["pack_cost"] == 0 else "paid")
    assert seen == set(captured)
    assert value["free_event_count"] == sum(float(event["pack_cost"]) == 0 for event in events)
    assert value["paid_event_count"] == sum(float(event["pack_cost"]) > 0 for event in events)
    first = value["first_case"]
    assert first["event"] == min(events, key=sort_key)
    for controller in ("dp30", "dp2"):
        record = first[controller]
        q = np.array(record["q_values"], float)
        assert q.shape == (7,) and np.all(np.isfinite(q))
        maximum = max(q)
        chosen = next(i for i in range(7) if q[i] >= maximum - 1e-12)
        assert chosen == record["action_index"]
        if controller == "dp30":
            assert chosen == first["event"]["executed_action"]
        same(record["planned_utility"], q[chosen], "purchase case selected Q")
        same(record["retire_q"], q[0], "purchase case retire Q")
        same(record["best_buy_minus_best_no_buy"], max(q[4:]) - max(q[:4]), "purchase versus no-buy Q gap")
        same(record["best_continue_minus_retire"], max(q[1:]) - q[0], "continue versus retire Q gap")
        selected_rows = [row for row in record["action_values"] if row["selected"]]
        assert len(selected_rows) == 1 and selected_rows[0]["action_index"] == chosen
        for i, row in enumerate(record["action_values"]):
            assert row["action_index"] == i
            same(row["q_value"], q[i], "purchase case action table")
    same(first["cash_spent_on_this_time_purchase"], first["event"]["pack_cost"], "purchase case cash spent")
    same(first["extra_trading_time_seconds"], 15., "purchase case added budget")
    return {"passed": True, "scope": "All post-hoc purchase events independently matched to unchanged dp30 ledgers; first chronological illustration, recorded state, selected Q, and Q-gap arithmetic verified. No full Bellman table was rebuilt for this addendum.",
            "events": len(events), "accounts_with_mid_purchase": len({account_key(e) for e in events}),
            "free_events": value["free_event_count"], "paid_events": value["paid_event_count"],
            "first_date": first["event"]["date"], "first_symbol": first["event"]["symbol"],
            "purchase_cases_sha256": digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--allow-partial", action="store_true", help="For an explicitly incomplete pilot only")
    parser.add_argument("--purchase-only", action="store_true", help="Append only the light post-hoc illustration check to the existing complete audit")
    args = parser.parse_args()
    if args.purchase_only:
        destination = ROOT / "verification.json"
        output = json.loads(destination.read_text(encoding="utf-8"))
        assert output["passed"] and not output["allow_partial_pilot"]
        assert output["summary_checks"]["results_sha256"] == digest(ROOT / "results.json")
        output["purchase_case_check"] = audit_purchase_cases()
        destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(output["purchase_case_check"], ensure_ascii=False))
        return
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    protocol_hashes = {name: digest(ROOT.parent / name) == wanted
                       for name, wanted in protocol["source_hashes"].items()}
    assert all(protocol_hashes.values()), "protocol source hashes changed"
    assert protocol["primary_pack_cost"] == 8 and protocol["primary_alpha"] == .1
    assert {float(s["pack_cost"]) for s in protocol["settings"]} == {0., 2., 8.}
    lock = json.loads((PRIOR / "lock_manifest.json").read_text(encoding="utf-8"))
    kernel_path = PRIOR / "kernel.json"
    kernel = json.loads(kernel_path.read_text(encoding="utf-8"))
    expected_kernel_hash = lock["frozen_artifacts"]["kernel.json"]
    hashes = {"frozen_kernel_unchanged": digest(kernel_path) == expected_kernel_hash}
    market = {}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        path = PRIOR / "data" / f"{symbol}-1d-2026.csv"
        hashes[symbol + "_source_unchanged"] = digest(path) == lock["holdout_sha256"][symbol]
        market[symbol] = load_market(path, kernel["volatility_threshold"])
    assert all(hashes.values()), hashes
    assert kernel["training_years"] == [2022, 2023, 2024, 2025]
    assert kernel["training_last_date"].startswith("2025-")
    directories = sorted(path.parent for path in args.runs.glob("*/ledger.csv.gz")
                         if (path.parent / "complete.json").exists())
    assert directories, f"no replay ledgers found in {args.runs}"
    if not args.allow_partial:
        expected_names = {f'cost{s["pack_cost"]:g}_alpha{s["alpha"]:g}_L{s["initial_life"]:g}' for s in protocol["settings"]}
        assert {path.name for path in directories} == expected_names, "full verification requires every predeclared setting"
    results = [audit_setting(directory, market, allow_partial=args.allow_partial) for directory in directories]
    summary_checks = {} if args.allow_partial else audit_summary(directories, market)
    output = {"verified_at": datetime.now(timezone.utc).isoformat(), "passed": True,
              "scope": "Independent complete transaction accounting, causal market-state reconstruction, frozen-data hashes, account reconciliation and independently recomputed aggregates. This is historical re-evaluation, not a new blind test, and not a proof of the planner's real-market optimality.",
              "allow_partial_pilot": args.allow_partial, "source_hashes_passed": hashes,
              "protocol_source_hashes_passed": protocol_hashes,
              "numeric_checks": COUNTS["scalar_numeric_checks"],
              "accounts": sum(r["accounts"] for r in results),
              "ledger_events": sum(r["ledger_events"] for r in results), "settings": results,
              "summary_checks": summary_checks}
    if not args.allow_partial and (ROOT / "purchase_cases.json").exists():
        output["purchase_case_check"] = audit_purchase_cases()
    destination = ROOT / ("verification_pilot.json" if args.allow_partial else "verification.json")
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("passed", "accounts", "ledger_events", "numeric_checks", "allow_partial_pilot")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
